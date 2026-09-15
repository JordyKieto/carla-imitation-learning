"""TFRecord formats.

Raw recording (drive.py / collect.py), one Example per 20 Hz frame:
    vision     uint8   (224, 224, 3)  RGB camera
    lidar      float32 (64, 64, 1)    LiDAR bird's-eye-view histogram, max-normalised
    telemetry  float32 (5,)           speed_mps, speed_kph, accel, v_forward, v_lateral (tanh-scaled)
    waypoints  float32 (25, 3)        per waypoint: dir_x, dir_y, distance / 50 (ego frame)
    controls   float32 (3,)           throttle, brake, steer (expert, smoothed)

Processed recording (preprocess.py): identical, but vision is resized to (96, 96, 3).
"""
import numpy as np
import tensorflow as tf

from utils.config import CONFIG

RAW_VISION_SHAPE = (CONFIG["sensors"]["camera"]["height"], CONFIG["sensors"]["camera"]["width"], 3)
VISION_SHAPE = tuple(CONFIG["data"]["vision_resize"]) + (3,)
LIDAR_SHAPE = (CONFIG["sensors"]["lidar"]["height"], CONFIG["sensors"]["lidar"]["width"], 1)
TELEMETRY_SHAPE = (5,)
WAYPOINTS_SHAPE = (CONFIG["waypoints"]["path_len"], 3)
CONTROLS_SHAPE = (3,)
CONTROL_NAMES = ("throttle", "brake", "steer")

FEATURES = {
    name: tf.io.FixedLenFeature([], tf.string)
    for name in ("vision", "lidar", "telemetry", "waypoints", "controls")
}


def _decode(ex, name, dtype, shape):
    return tf.reshape(tf.io.decode_raw(ex[name], dtype), shape)


def parse_raw(example_proto):
    """Raw frame -> (vision uint8 resized to 96x96, lidar, telemetry, waypoints, controls)."""
    ex = tf.io.parse_single_example(example_proto, FEATURES)
    vision = _decode(ex, "vision", tf.uint8, RAW_VISION_SHAPE)
    vision = tf.image.resize(tf.cast(vision, tf.float32), VISION_SHAPE[:2])
    vision = tf.cast(tf.clip_by_value(tf.round(vision), 0.0, 255.0), tf.uint8)
    return (
        vision,
        _decode(ex, "lidar", tf.float32, LIDAR_SHAPE),
        _decode(ex, "telemetry", tf.float32, TELEMETRY_SHAPE),
        _decode(ex, "waypoints", tf.float32, WAYPOINTS_SHAPE),
        _decode(ex, "controls", tf.float32, CONTROLS_SHAPE),
    )


def parse_processed(example_proto):
    """Processed frame -> ((vision float [0,1], lidar, telemetry, waypoints), controls)."""
    ex = tf.io.parse_single_example(example_proto, FEATURES)
    vision = tf.cast(_decode(ex, "vision", tf.uint8, VISION_SHAPE), tf.float32) / 255.0
    inputs = (
        vision,
        _decode(ex, "lidar", tf.float32, LIDAR_SHAPE),
        _decode(ex, "telemetry", tf.float32, TELEMETRY_SHAPE),
        _decode(ex, "waypoints", tf.float32, WAYPOINTS_SHAPE),
    )
    return inputs, _decode(ex, "controls", tf.float32, CONTROLS_SHAPE)


def serialize_processed(vision, lidar, telemetry, waypoints, controls):
    def feature(array, dtype):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.asarray(array, dtype).tobytes()]))

    example = tf.train.Example(features=tf.train.Features(feature={
        "vision": feature(vision, np.uint8),
        "lidar": feature(lidar, np.float32),
        "telemetry": feature(telemetry, np.float32),
        "waypoints": feature(waypoints, np.float32),
        "controls": feature(controls, np.float32),
    }))
    return example.SerializeToString()
