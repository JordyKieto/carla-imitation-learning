"""Window manifest: one Parquet row per trainable stack of consecutive frames.

A window is `stack_size` consecutive frames of one processed recording. Its
target is the last frame's controls. Each frame gets a direction label from
its steer value; windows whose frames disagree too much are dropped.
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.records import parse_processed
from utils.config import CONFIG

STACK_SIZE = int(CONFIG["data"]["stack_size"])
STEER_THRESHOLD = float(CONFIG["data"]["direction_steer_threshold"])
MIN_DIRECTION_SHARE = float(CONFIG["data"]["window_min_direction_share"])
DIRECTION_NAMES = ("straight", "left", "right")
DIRECTION_TO_ID = {name: idx for idx, name in enumerate(DIRECTION_NAMES)}


def direction_from_steer(steer):
    if steer <= -STEER_THRESHOLD:
        return "left"
    if steer >= STEER_THRESHOLD:
        return "right"
    return "straight"


def frame_quality(vision, lidar, telemetry, waypoints, controls, vision_shape):
    flags = []
    for name, array in (("telemetry", telemetry), ("waypoints", waypoints),
                        ("controls", controls), ("lidar", lidar)):
        if not np.all(np.isfinite(array)):
            flags.append(f"bad_{name}")
    if tuple(vision.shape) != tuple(vision_shape):
        flags.append("bad_vision_shape")
    if np.max(np.abs(controls)) > 5.0:
        flags.append("control_outlier")
    return "ok" if not flags else "|".join(flags)


def window_rows(frames, processed_file, source_file, tags):
    """frames: list of per-frame dicts with keys steer/throttle/brake/speed/direction/quality."""
    rows = []
    for start in range(0, len(frames) - STACK_SIZE + 1):
        window = frames[start:start + STACK_SIZE]
        dirs = [f["direction"] for f in window]
        direction = max(DIRECTION_NAMES, key=dirs.count)
        share = dirs.count(direction) / float(STACK_SIZE)
        if share < MIN_DIRECTION_SHARE:
            continue
        bad = sorted({f["quality"] for f in window if f["quality"] != "ok"})
        target = window[-1]
        rows.append({
            "processed_file": processed_file,
            "source_file": source_file,
            "start_index": start,
            "direction": direction,
            "direction_share": share,
            "tags": list(tags),
            "quality_flags": "ok" if not bad else "|".join(bad),
            "target_throttle": target["throttle"],
            "target_brake": target["brake"],
            "target_steer": target["steer"],
            "target_speed": target["speed"],
        })
    return rows


def filter_manifest(df, tags=()):
    df = df[df["quality_flags"] == "ok"]
    if tags:
        wanted = set(tags)
        df = df[df["tags"].map(lambda values: wanted.issubset(set(values)))]
    if df.empty:
        raise RuntimeError("No windows left after filtering. Check tags / manifest contents.")
    return df


def split_train_val(df, train_split):
    """Per (recording, direction): first `train_split` of windows -> train, rest -> val."""
    df = df.sort_values(["processed_file", "direction", "start_index"]).reset_index(drop=True)
    position = df.groupby(["processed_file", "direction"]).cumcount()
    size = df.groupby(["processed_file", "direction"])["start_index"].transform("size")
    n_train = size.map(lambda n: int(round(n * float(train_split))))
    train, val = df[position < n_train], df[position >= n_train]
    if train.empty:
        train = df
    if val.empty:
        val = train
    return train, val


def to_ranges(df):
    """Collapses consecutive windows of the same file/direction into (file, start, count, direction)."""
    df = df.sort_values(["processed_file", "direction", "start_index"])
    new_run = (
        (df["processed_file"] != df["processed_file"].shift())
        | (df["direction"] != df["direction"].shift())
        | (df["start_index"] != df["start_index"].shift() + 1)
    ).cumsum()
    grouped = df.groupby(new_run)
    return pd.DataFrame({
        "processed_file": grouped["processed_file"].first(),
        "start": grouped["start_index"].first(),
        "num_windows": grouped.size(),
        "direction": grouped["direction"].first(),
    }).reset_index(drop=True)


def make_dataset(ranges, processed_dir, batch_size, shuffle):
    """Yields ((vision, lidar, telemetry, waypoints) stacks, controls[-1], direction_id) batches."""

    def windows_for_range(path, start, num_windows, direction_id):
        ds = tf.data.TFRecordDataset(path).skip(start).take(num_windows + STACK_SIZE - 1)
        ds = ds.map(parse_processed, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.window(STACK_SIZE, shift=1, drop_remainder=True)

        def stack(inputs, controls):
            return tf.data.Dataset.zip((
                tf.data.Dataset.zip(inputs).batch(STACK_SIZE, drop_remainder=True),
                controls.batch(STACK_SIZE, drop_remainder=True),
            ))

        ds = ds.flat_map(stack)
        return ds.map(lambda inputs, controls: (inputs, controls[-1], direction_id))

    paths = [str(processed_dir / name) for name in ranges["processed_file"]]
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(paths),
        tf.constant(ranges["start"].to_numpy(np.int64)),
        tf.constant(ranges["num_windows"].to_numpy(np.int64)),
        tf.constant([DIRECTION_TO_ID[d] for d in ranges["direction"]], dtype=tf.int64),
    ))
    if shuffle:
        ds = ds.shuffle(len(ranges))
    ds = ds.interleave(
        windows_for_range,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
