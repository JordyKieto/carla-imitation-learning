"""Drive in CARLA: manual control, autopilot, recording, and policy inference.

    python drive.py [MAP] [POLICY_PATH]

Controls: Arrows=drive | P=autopilot | R=start/stop recording | K=run policy
          G=DAgger | J=RL fine-tune | I/U=next/prev spawn | M/N=next/prev map
          B=minimap | Y=reverse | ESC=quit
Recordings are written to recordings/drive_<timestamp>.tfrecord (+ .json summary).
"""
import carla
import pygame
import numpy as np
import collections
import sys
import logging
import random
import time
import math
import os
import json
import shutil
import tempfile
import zipfile

import tensorflow as tf

from utils.config import env
from vehicle.ego_context import EgoContext
from vehicle.vehicle_reward import VehicleReward
from vehicle.vehicle_sensors import VehicleSensors
from vehicle.vehicle_telemetry import VehicleTelemetry
from vehicle.vehicle_waypoints import Waypoints

# ============================
# Config
# ============================
DATA_DIR = "./recordings"
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
STACK_SIZE = 5
INFER_LOG_EVERY = 10
S3_MODEL_CACHE_DIR = "checkpoints/inference"
VISION_RESIZE = (96, 96)

# ============================
# Configuration
# ============================
CONFIG = {
    "host": "localhost",
    "port": 2000,
    "traffic_manager_port": 8000,
    "width": 1280,
    "height": 720,
    "synchronous_mode": True,
    "vehicle_filter": "vehicle.tesla.model3",
    "camera": {
        "x": -8.0,
        "z": 3.5,
        "pitch": -10.0,
        "fov": "90",
    },
    "record": False,
    "record_dir": "recordings",
    "inference": False,
    "inference_model_path": "artifacts/policy.keras",
    "always_green_lights": False,
    "background_agents": {
        "vehicles": {
            "count": 0,
            "filter": "vehicle.*",
        },
        "pedestrians": {
            "count": 0,
            "filter": "walker.pedestrian.*",
            "speed_min": 1.0,
            "speed_max": 2.0,
        },
        "animals": {
            "count": 0,
            "filter": "walker.animal.*",
            "speed_min": 0.5,
            "speed_max": 1.5,
        },
    },
    "dagger": {
        "window": 30,
        "infer_steps": 15,
        "autopilot_steps": 15,
        "checkpoint_every": 1000,
        "loss_ema": 0.95,
        "loss_log_every": 100,
    },
    "reinforcement": {
        "rollout_steps": 32,
        "gamma": 0.98,
        "action_noise": [0.08, 0.04, 0.12],
        "reward_temperature": 2.0,
        "reward_ema": 0.95,
        "reward_log_every": 50,
        "checkpoint_every": 1000,
    },
}

# ============================
# Policy model (load / save for inference, DAgger and RL modes)
# ============================
class MLPTrainer:
    def __init__(self):
        self.model = None

    @staticmethod
    def _strip_keras_compat_fields(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if key == "quantization_config":
                    continue
                cleaned[key] = MLPTrainer._strip_keras_compat_fields(value)
            return cleaned
        if isinstance(obj, list):
            return [MLPTrainer._strip_keras_compat_fields(v) for v in obj]
        return obj

    def _load_model_compat(self, path):
        suffix = os.path.splitext(path)[1].lower()
        if suffix != ".keras":
            raise

        temp_dir = tempfile.mkdtemp(prefix="keras_compat_")
        try:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(temp_dir)

            config_path = os.path.join(temp_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Missing config.json in model archive: {path}")

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config = self._strip_keras_compat_fields(config)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)

            compat_path = os.path.join(temp_dir, "compat_model.keras")
            with zipfile.ZipFile(compat_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(temp_dir):
                    for name in files:
                        full_path = os.path.join(root, name)
                        if full_path == compat_path:
                            continue
                        arcname = os.path.relpath(full_path, temp_dir)
                        zf.write(full_path, arcname)

            return tf.keras.models.load_model(compat_path, safe_mode=False, compile=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _parse_s3_uri(uri):
        without_scheme = uri[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {uri}")
        return bucket, key

    def _local_model_path(self, path):
        if not str(path).startswith("s3://"):
            return path

        bucket, key = self._parse_s3_uri(path)
        os.makedirs(S3_MODEL_CACHE_DIR, exist_ok=True)
        local_path = os.path.join(S3_MODEL_CACHE_DIR, os.path.basename(key))
        if not os.path.exists(local_path):
            logger.info("[MODEL] Downloading %s -> %s", path, local_path)
            import boto3  # credentials come from .env / environment

            boto3.client("s3", endpoint_url=env("S3_ENDPOINT_URL")).download_file(bucket, key, local_path)
        return local_path

    @staticmethod
    def _control_output(outputs):
        if isinstance(outputs, dict):
            return outputs["controls"]
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs
    
    def save(self, path):
        if self.model is None:
            raise ValueError("Model is not built yet.")
        self.model.save(path)

    def load(self, path):
        path = self._local_model_path(path)
        try:
            self.model = tf.keras.models.load_model(path, safe_mode=False, compile=False)
        except TypeError as exc:
            if "quantization_config" not in str(exc):
                raise
            logger.warning("[MODEL] Retrying load with Keras compatibility sanitization: %s", path)
            self.model = self._load_model_compat(path)
        return self.model

# ============================
# Logging Setup
# ============================
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CarlaApp")


class DrivingInputContainer(EgoContext, VehicleSensors, VehicleTelemetry, Waypoints, VehicleReward):
    def __init__(self, world, vehicle, control_smoothing=None):
        super().__init__(
            world=world,
            vehicle=vehicle,
            enable_camera=True,
            enable_semantic=False,
            enable_lidar=True,
            enable_collision=True,
        )
        self._telemetry_spec = {
            "speed_mps": True,
            "speed_kph": True,
            "accel_mps2": True,
            "v_forward": True,
            "v_lateral": True,
        }
        self._telemetry_layout = [
            "speed_mps",
            "speed_kph",
            "accel_mps2",
            "v_forward",
            "v_lateral",
        ]
        self._reward_telemetry_spec = {
            "speed_mps": True,
            "speed_kph": True,
            "accel_mps2": True,
            "v_forward": True,
            "v_lateral": True,
            "steering_angle": True,
            "throttle_input": True,
            "brake_input": True,
        }
        self._reward_telemetry_layout = [
            "speed_mps",
            "speed_kph",
            "accel_mps2",
            "v_forward",
            "v_lateral",
            "steering_angle",
            "throttle_input",
            "brake_input",
        ]
        self._brake_index = None

        self._control_smoothing = {
            "enabled": True,
            "dt": None,
            "throttle_tau": 0.5,
            "throttle_rate_up": 1.0,
            "throttle_rate_down": 1.5,
            "throttle_deadband": 0.03,
            "steer_tau": 0.2,
            "steer_rate": 2.5,
        }
        if control_smoothing:
            self._control_smoothing.update(control_smoothing)

        self._throttle_lp = None
        self._throttle_prev = None
        self._steer_lp = None
        self._steer_prev = None

    def _resolve_dt(self, snap):
        cfg_dt = self._control_smoothing.get("dt")
        if cfg_dt is not None:
            return float(cfg_dt)
        try:
            ts = getattr(snap, "timestamp", None)
            if ts is not None and hasattr(ts, "delta_seconds"):
                dt = float(ts.delta_seconds)
                if dt > 0.0:
                    return dt
        except Exception:
            pass
        try:
            settings = self.world.get_settings()
            dt = float(getattr(settings, "fixed_delta_seconds", 0.0))
            if dt > 0.0:
                return dt
        except Exception:
            pass
        return 0.05

    @staticmethod
    def _ema_step(raw, prev, tau, dt):
        if prev is None:
            return raw
        if tau <= 0.0:
            return raw
        alpha = dt / (tau + dt)
        return (alpha * raw) + ((1.0 - alpha) * prev)

    @staticmethod
    def _rate_limit_step(target, prev, rate_up, rate_down, dt):
        if prev is None:
            return target
        delta = target - prev
        max_up = rate_up * dt
        max_down = rate_down * dt
        if delta > max_up:
            delta = max_up
        elif delta < -max_down:
            delta = -max_down
        return prev + delta

    @staticmethod
    def _deadband(value, eps):
        return 0.0 if abs(value) < eps else value

    def _smooth_throttle(self, raw, dt):
        cfg = self._control_smoothing
        if not cfg.get("enabled", True):
            self._throttle_lp = raw
            self._throttle_prev = raw
            return raw

        self._throttle_lp = self._ema_step(
            raw, self._throttle_lp, float(cfg["throttle_tau"]), dt
        )
        smoothed = self._rate_limit_step(
            self._throttle_lp,
            self._throttle_prev,
            float(cfg["throttle_rate_up"]),
            float(cfg["throttle_rate_down"]),
            dt,
        )
        smoothed = self._deadband(smoothed, float(cfg["throttle_deadband"]))
        self._throttle_prev = smoothed
        return smoothed

    def _smooth_steer(self, raw, dt):
        cfg = self._control_smoothing
        if not cfg.get("enabled", True):
            self._steer_lp = raw
            self._steer_prev = raw
            return raw

        self._steer_lp = self._ema_step(
            raw, self._steer_lp, float(cfg["steer_tau"]), dt
        )
        smoothed = self._rate_limit_step(
            self._steer_lp,
            self._steer_prev,
            float(cfg["steer_rate"]),
            float(cfg["steer_rate"]),
            dt,
        )
        self._steer_prev = smoothed
        return smoothed

    def collect_step(self, include_reward=False):
        snap = self.snapshot()
        if snap is None:
            return None
        rgb_u8, _, lidar_f32, collision, _, _ = snap
        telem_vec = self._compute_telemetry(self._telemetry_spec, self._telemetry_layout)
        wp_norm, wp_raw = self._future_waypoint_path()
        ctrl = self.vehicle.get_control()
        dt = self._resolve_dt(snap)
        throttle = float(self._smooth_throttle(float(ctrl.throttle), dt))
        steer = float(self._smooth_steer(float(ctrl.steer), dt))
        brake = float(ctrl.brake)
        controls = np.array([throttle, brake, steer], dtype=np.float32)
        if not include_reward:
            return rgb_u8, lidar_f32[..., None], telem_vec, wp_norm, controls

        reward_telem = self._compute_telemetry(
            self._reward_telemetry_spec, self._reward_telemetry_layout
        )
        reward = float(self.compute_reward(float(collision), reward_telem, wp_raw))
        return rgb_u8, lidar_f32[..., None], telem_vec, wp_norm, controls, reward, float(collision)

    def pop_collision(self):
        collision = float(self.get_collision())
        if collision > 0.0:
            self.clear_collision()
        return collision


class TFRecordDriveRecorder:
    STEER_THRESH = 0.15

    def __init__(self, record_dir):
        self.record_dir = record_dir
        self.recording = False
        self.writer = None
        self.record_path = None
        self.map_name = None
        self.spawn_index = None
        self.started_at = None
        self.schema = None
        self._reset_stats()

    def _reset_stats(self):
        self.total_steps = 0
        self.total_brakes = 0
        self.turn_counts = {"left": 0, "straight": 0, "right": 0}
        self.schema = None

    def start(self, map_name, spawn_index):
        if self.recording:
            return
        os.makedirs(self.record_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.record_path = os.path.join(self.record_dir, f"drive_{stamp}.tfrecord")
        self.writer = tf.io.TFRecordWriter(self.record_path)
        self.recording = True
        self.map_name = map_name
        self.spawn_index = int(spawn_index)
        self.started_at = time.time()
        self._reset_stats()
        logger.info(f"[RECORD] Started: {self.record_path}")

    def stop(self, alignment_stats=None):
        if not self.recording:
            return None
        if self.writer is not None:
            self.writer.close()
        self.writer = None
        self.recording = False
        summary = self._build_summary(alignment_stats=alignment_stats)
        summary_path = os.path.splitext(self.record_path)[0] + ".json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[RECORD] Stopped: {self.record_path}")
        logger.info(
            f"[RECORD] Summary: {summary_path} | steps={summary['steps']} "
            f"map={summary['map']} spawn_idx={summary['spawn_index']}"
        )
        ratios = summary["control_ratios"]
        logger.info(
            "[RECORD] ratios (%) | "
            f"left={ratios['left']:.1f} straight={ratios['straight']:.1f} "
            f"right={ratios['right']:.1f}"
        )
        logger.info(f"[RECORD] brakes | total={summary['total_brakes']}")
        return summary

    def append(self, vision, lidar, telemetry, waypoints, controls):
        if not self.recording or self.writer is None:
            return
        brake = float(controls[1])
        steer = float(controls[2])
        if steer < -self.STEER_THRESH:
            turn = "left"
        elif steer > self.STEER_THRESH:
            turn = "right"
        else:
            turn = "straight"
        if brake > 0.0:
            self.total_brakes += 1
        self.turn_counts[turn] += 1
        self.total_steps += 1

        vision = np.ascontiguousarray(vision)
        lidar = np.ascontiguousarray(lidar)
        telemetry = np.ascontiguousarray(telemetry)
        waypoints = np.ascontiguousarray(waypoints)
        controls = np.ascontiguousarray(controls)
        if self.schema is None:
            self.schema = {
                "vision": {"dtype": str(vision.dtype), "shape": list(vision.shape)},
                "lidar": {"dtype": str(lidar.dtype), "shape": list(lidar.shape)},
                "telemetry": {"dtype": str(telemetry.dtype), "shape": list(telemetry.shape)},
                "waypoints": {"dtype": str(waypoints.dtype), "shape": list(waypoints.shape)},
                "controls": {"dtype": str(controls.dtype), "shape": list(controls.shape)},
                "encoding": "raw_bytes",
            }

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "vision": _bytes_feature(vision.tobytes()),
                    "vision_shape": _int64_list(vision.shape),
                    "lidar": _bytes_feature(lidar.tobytes()),
                    "lidar_shape": _int64_list(lidar.shape),
                    "telemetry": _bytes_feature(telemetry.tobytes()),
                    "telemetry_shape": _int64_list(telemetry.shape),
                    "waypoints": _bytes_feature(waypoints.tobytes()),
                    "waypoints_shape": _int64_list(waypoints.shape),
                    "controls": _bytes_feature(controls.tobytes()),
                    "controls_shape": _int64_list(controls.shape),
                }
            )
        )
        self.writer.write(example.SerializeToString())

    def _build_summary(self, alignment_stats=None):
        total_turns = sum(self.turn_counts.values())
        if total_turns > 0:
            ratios = {
                "left": (self.turn_counts["left"] / total_turns) * 100.0,
                "straight": (self.turn_counts["straight"] / total_turns) * 100.0,
                "right": (self.turn_counts["right"] / total_turns) * 100.0,
            }
        else:
            ratios = {
                "left": 0.0,
                "straight": 0.0,
                "right": 0.0,
            }
        return {
            "steps": self.total_steps,
            "map": self.map_name,
            "spawn_index": self.spawn_index,
            "record_path": self.record_path,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "total_brakes": self.total_brakes,
            "control_ratios": ratios,
            "alignment_stats": alignment_stats or {},
            "tfrecord_schema": self.schema or {},
        }


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


class CarlaCameraApp(Waypoints):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.client = None
        self.world = None
        self.tm = None
        self.vehicle = None
        self.camera = None

        # Map cycling
        self.available_maps = []
        self.map_index = 0
        self.spawn_points = []
        self.spawn_index = 0

        # Vehicle control
        self.control = carla.VehicleControl()
        self.in_reverse = False
        self.autopilot_enabled = False
        self.inference_enabled = False
        self.inference_model_path = self.cfg.get(
            "inference_model_path", "./mlp_controls.keras"
        )
        self.inference_trainer = MLPTrainer()
        self.inference_model = None
        dagger_cfg = self.cfg.get("dagger", {})
        self.dagger_window = int(dagger_cfg.get("window", 10))
        self.dagger_infer_steps = int(dagger_cfg.get("infer_steps", 5))
        self.dagger_autopilot_steps = int(
            dagger_cfg.get("autopilot_steps", max(0, self.dagger_window - self.dagger_infer_steps))
        )
        if self.dagger_infer_steps + self.dagger_autopilot_steps <= 0:
            self.dagger_infer_steps = 5
            self.dagger_autopilot_steps = 5
        if self.dagger_infer_steps + self.dagger_autopilot_steps != self.dagger_window:
            self.dagger_window = self.dagger_infer_steps + self.dagger_autopilot_steps
        self.dagger_checkpoint_every = int(dagger_cfg.get("checkpoint_every", 1000))
        self.dagger_enabled = False
        self.dagger_steps = 0
        self.dagger_loss_ema = None
        self.dagger_loss_alpha = float(dagger_cfg.get("loss_ema", 0.95))
        self.dagger_loss_log_every = int(dagger_cfg.get("loss_log_every", 100))
        self.dagger_batch = []
        self.dagger_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.dagger_recorder = TFRecordDriveRecorder(DATA_DIR)
        self.dagger_controller = "model"
        self.dagger_window_step = 0
        rl_cfg = self.cfg.get("reinforcement", {})
        self.rl_enabled = False
        self.rl_steps = 0
        self.rl_rollout_steps = int(rl_cfg.get("rollout_steps", 32))
        self.rl_gamma = float(rl_cfg.get("gamma", 0.98))
        self.rl_action_noise = np.asarray(
            rl_cfg.get("action_noise", [0.08, 0.04, 0.12]), dtype=np.float32
        )
        if self.rl_action_noise.shape != (3,):
            self.rl_action_noise = np.asarray([0.08, 0.04, 0.12], dtype=np.float32)
        self.rl_reward_temperature = max(1e-4, float(rl_cfg.get("reward_temperature", 2.0)))
        self.rl_reward_alpha = float(rl_cfg.get("reward_ema", 0.95))
        self.rl_reward_log_every = int(rl_cfg.get("reward_log_every", 50))
        self.rl_checkpoint_every = int(rl_cfg.get("checkpoint_every", 1000))
        self.rl_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE * 0.5)
        self.rl_reward_ema = None
        self.rl_rollout = []
        self.rl_pending = None
        self.rl_collision_cooldown = 0

        # Pygame
        self.display = None
        self.clock = None
        self.image_surface = None
        self.latest_image = None
        self.latest_image_dims = None
        self.running = True
        self.frame_count = 0
        self.minimap_surface = None
        self.minimap_size = (220, 220)
        self.minimap_padding = 10
        self.minimap_bounds = None
        self.minimap_enabled = True
        self.data_collector = None
        self.recorder = TFRecordDriveRecorder(self.cfg["record_dir"])
        self.record_on_start = bool(self.cfg.get("record", False))
        self.inference_on_start = bool(self.cfg.get("inference", False))
        self.background_vehicles = []
        self.background_walkers = []
        self.background_walker_controllers = []
        self.active_waypoint_route = None
        self._vision_history = collections.deque(maxlen=STACK_SIZE)
        self._lidar_history = collections.deque(maxlen=STACK_SIZE)
        self._telemetry_history = collections.deque(maxlen=STACK_SIZE)
        self._waypoint_history = collections.deque(maxlen=STACK_SIZE)
        self._infer_log_step = 0

    # ============================
    # Initialization
    # ============================
    def initialize(self):
        self._init_pygame()
        self._connect_client()
        self._load_maps()
        self._load_world(self.cfg["map"])
        self._spawn_vehicle()
        self._spawn_background_agents()
        self._setup_camera()
        self._setup_data_collector()
        self._log_controls()
        if self.record_on_start:
            self.start_recording()
        if self.inference_on_start:
            self._set_inference_enabled(True)
        logger.info("Initialization complete")

    def _init_pygame(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.cfg["width"], self.cfg["height"]),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        pygame.display.set_caption("CARLA Manual Control + Map Cycle")
        self.clock = pygame.time.Clock()

    def _connect_client(self):
        self.client = carla.Client(self.cfg["host"], self.cfg["port"])
        self.client.set_timeout(30.0)
        self.tm = self.client.get_trafficmanager(self.cfg["traffic_manager_port"])
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        try:
            self.tm.set_respawn_dormant_vehicles(True)
        except Exception:
            pass
        self.tm.set_synchronous_mode(bool(self.cfg.get("synchronous_mode", False)))

    def _load_maps(self):
        self.available_maps = sorted(
            [m for m in self.client.get_available_maps() if "Town" in m]
        )
        print(f"Available maps: {self.available_maps}" )
        if not self.available_maps:
            raise RuntimeError("No CARLA maps found")

    # ============================
    # World / Actor Lifecycle
    # ============================
    def _load_world(self, map_name):
        logger.info(f"[CARLA] Loading map: {map_name}")
        self.client.load_world(map_name)
        self.world = self.client.get_world()

        sync_enabled = bool(self.cfg.get("synchronous_mode", False))
        settings = self.world.get_settings()
        settings.synchronous_mode = sync_enabled
        if sync_enabled and not settings.fixed_delta_seconds:
            settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        if self.tm:
            self.tm.set_synchronous_mode(sync_enabled)

        # Give CARLA a moment to fully initialize the episode
        time.sleep(0.5)
        if self.cfg.get("always_green_lights", True):
            self._set_all_traffic_lights_green()
        self.spawn_points = self.world.get_map().get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available")
        self.spawn_index = 0
        self._build_minimap()

    def _spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find(self.cfg["vehicle_filter"])

        transform = self.spawn_points[self.spawn_index]

        self.vehicle = self.world.spawn_actor(vehicle_bp, transform)
        self.vehicle.set_autopilot(False)
        if self.tm:
            self.tm.global_percentage_speed_difference(60)
            self.tm.set_global_distance_to_leading_vehicle(2.0)
        #     self.tm.ignore_lights_percentage(self.vehicle, 100.0)
        #     self.tm.ignore_signs_percentage(self.vehicle, 100.0)

        logger.info(
            f"[SPAWN] Vehicle id={self.vehicle.id} on {self.world.get_map().name} | Spawn idx={self.spawn_index}"
        )

    def _spawn_background_agents(self):
        cfg = self.cfg.get("background_agents", {})
        self._spawn_background_vehicles(cfg.get("vehicles", {}))
        self._spawn_background_walkers(cfg.get("pedestrians", {}))
        self._spawn_background_walkers(cfg.get("animals", {}))

    def _spawn_background_vehicles(self, cfg):
        count = int(cfg.get("count", 0))
        if count <= 0:
            return
        bp_filter = cfg.get("filter", "vehicle.*")
        bp_lib = self.world.get_blueprint_library()
        blueprints = bp_lib.filter(bp_filter)
        if not blueprints:
            logger.info(f"[BG] No vehicle blueprints for filter='{bp_filter}'")
            return

        spawn_points = list(self.spawn_points)
        random.shuffle(spawn_points)
        for sp in spawn_points:
            if len(self.background_vehicles) >= count:
                break
            if self.vehicle and sp.location.distance(self.vehicle.get_transform().location) < 2.0:
                continue
            bp = random.choice(blueprints)
            actor = self.world.try_spawn_actor(bp, sp)
            if actor is None:
                continue
            if self.tm:
                actor.set_autopilot(True, self.tm.get_port())
            else:
                actor.set_autopilot(True)
            self.background_vehicles.append(actor)

        logger.info(f"[BG] Spawned {len(self.background_vehicles)}/{count} vehicles")

    def _spawn_background_walkers(self, cfg):
        count = int(cfg.get("count", 0))
        if count <= 0:
            return
        bp_filter = cfg.get("filter", "walker.pedestrian.*")
        speed_min = float(cfg.get("speed_min", 1.0))
        speed_max = float(cfg.get("speed_max", 2.0))
        if speed_max < speed_min:
            speed_max = speed_min

        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter(bp_filter)
        if not walker_bps:
            logger.info(f"[BG] No walker blueprints for filter='{bp_filter}'")
            return
        try:
            controller_bp = bp_lib.find("controller.ai.walker")
        except Exception:
            logger.info("[BG] Missing controller.ai.walker blueprint")
            return

        spawned = 0
        for _ in range(count):
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            transform = carla.Transform(location)
            walker_bp = random.choice(walker_bps)
            walker = self.world.try_spawn_actor(walker_bp, transform)
            if walker is None:
                continue
            controller = self.world.spawn_actor(
                controller_bp, carla.Transform(), attach_to=walker
            )
            controller.start()
            target = self.world.get_random_location_from_navigation() or location
            controller.go_to_location(target)
            controller.set_max_speed(random.uniform(speed_min, speed_max))
            self.background_walkers.append(walker)
            self.background_walker_controllers.append(controller)
            spawned += 1

        logger.info(f"[BG] Spawned {spawned}/{count} walkers for filter='{bp_filter}'")

    def _setup_camera(self):
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")

        camera_bp.set_attribute("image_size_x", str(self.cfg["width"]))
        camera_bp.set_attribute("image_size_y", str(self.cfg["height"]))
        camera_bp.set_attribute("fov", self.cfg["camera"]["fov"])

        transform = carla.Transform(
            carla.Location(
                x=self.cfg["camera"]["x"], z=self.cfg["camera"]["z"]
            ),
            carla.Rotation(pitch=self.cfg["camera"]["pitch"]),
        )

        self.camera = self.world.spawn_actor(
            camera_bp,
            transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.SpringArmGhost,
        )

        self.camera.listen(self._on_image)

    def _setup_data_collector(self):
        if self.data_collector is not None:
            try:
                self.data_collector.destroy()
            except Exception:
                pass
        self.data_collector = DrivingInputContainer(
            world=self.world,
            vehicle=self.vehicle,
        )
        if self.active_waypoint_route is not None:
            self.data_collector.set_waypoint_route(self.active_waypoint_route)
        self._vision_history.clear()
        self._lidar_history.clear()
        self._telemetry_history.clear()
        self._waypoint_history.clear()

    def set_active_waypoint_route(self, route):
        self.active_waypoint_route = list(route) if route else None
        if self.data_collector is not None:
            if self.active_waypoint_route is None:
                self.data_collector.clear_waypoint_route()
            else:
                self.data_collector.set_waypoint_route(self.active_waypoint_route)

    def _stack_history(self, history, current_value):
        values = list(history)
        if not values:
            values = [current_value]
        while len(values) < STACK_SIZE:
            values.insert(0, values[0])
        return np.stack(values[-STACK_SIZE:], axis=0)

    @staticmethod
    def _preprocess_model_inputs(vision, lidar, telemetry, waypoints):
        vision = tf.cast(vision, tf.float32) / 255.0
        if VISION_RESIZE is not None:
            vision = tf.image.resize(vision, VISION_RESIZE)
        return (
            np.asarray(vision.numpy(), dtype=np.float32),
            np.asarray(lidar, dtype=np.float32),
            np.asarray(telemetry, dtype=np.float32),
            np.asarray(waypoints, dtype=np.float32),
        )

    def _build_minimap(self):
        carla_map = self.world.get_map()
        topology = carla_map.get_topology()
        if not topology:
            self.minimap_surface = None
            self.minimap_bounds = None
            return

        points = []
        for wp_start, wp_end in topology:
            points.append((wp_start.transform.location.x, wp_start.transform.location.y))
            points.append((wp_end.transform.location.x, wp_end.transform.location.y))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 5.0
        self.minimap_bounds = (min_x - pad, max_x + pad, min_y - pad, max_y + pad)

        self.minimap_surface = pygame.Surface(self.minimap_size)
        self.minimap_surface.fill((15, 15, 18))
        road_color = (120, 120, 120)
        for wp_start, wp_end in topology:
            p0 = self._world_to_minimap(wp_start.transform.location)
            p1 = self._world_to_minimap(wp_end.transform.location)
            if p0 and p1:
                pygame.draw.line(self.minimap_surface, road_color, p0, p1, 2)

    def _set_all_traffic_lights_green(self):
        if self.world is None:
            return
        try:
            lights = self.world.get_actors().filter("traffic.traffic_light*")
        except Exception:
            return
        for light in lights:
            try:
                light.set_state(carla.TrafficLightState.Green)
                light.set_green_time(9999.0)
            except Exception:
                pass

    def _world_to_minimap(self, location):
        if self.minimap_bounds is None:
            return None
        min_x, max_x, min_y, max_y = self.minimap_bounds
        width, height = self.minimap_size
        pad = self.minimap_padding
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x <= 0 or span_y <= 0:
            return None
        x = (location.x - min_x) / span_x
        y = (location.y - min_y) / span_y
        px = int(pad + x * (width - 2 * pad))
        py = int(pad + (1.0 - y) * (height - 2 * pad))
        return (px, py)

    def _draw_minimap(self):
        if not self.minimap_enabled or self.vehicle is None:
            return
        if self.minimap_surface is None:
            self._build_minimap()
        if self.minimap_surface is None:
            return
        overlay = self.minimap_surface.copy()
        self._draw_minimap_actor(overlay, self.vehicle, (40, 220, 120), radius=4, heading_len=12)
        for actor in self.background_vehicles:
            self._draw_minimap_actor(overlay, actor, (90, 160, 255), radius=3, heading_len=8)
        for actor in self.background_walkers:
            self._draw_minimap_actor(overlay, actor, (255, 190, 80), radius=3, heading_len=6)

        pos_x = self.cfg["width"] - self.minimap_size[0] - 10
        pos_y = self.cfg["height"] - self.minimap_size[1] - 10
        self.display.blit(overlay, (pos_x, pos_y))

    def _draw_minimap_actor(self, surface, actor, color, radius=3, heading_len=8):
        if actor is None:
            return
        try:
            tf = actor.get_transform()
        except Exception:
            return
        pt = self._world_to_minimap(tf.location)
        if not pt:
            return
        pygame.draw.circle(surface, color, (int(pt[0]), int(pt[1])), radius)
        heading = math.radians(-tf.rotation.yaw)
        dx = math.cos(heading) * heading_len
        dy = math.sin(heading) * heading_len
        end = (int(pt[0] + dx), int(pt[1] + dy))
        pygame.draw.line(surface, color, (int(pt[0]), int(pt[1])), end, 2)

    def cleanup_actors(self):
        if self.data_collector is not None:
            try:
                self.data_collector.destroy()
            except Exception:
                pass
            self.data_collector = None
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None

        for controller in self.background_walker_controllers:
            try:
                controller.stop()
                controller.destroy()
            except Exception:
                pass
        self.background_walker_controllers = []

        for walker in self.background_walkers:
            try:
                walker.destroy()
            except Exception:
                pass
        self.background_walkers = []

        for vehicle in self.background_vehicles:
            try:
                vehicle.destroy()
            except Exception:
                pass
        self.background_vehicles = []

        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

        # Critical: allow CARLA to flush callbacks
        time.sleep(0.3)

    # ============================
    # Input / Control
    # ============================
    def _handle_controls(self):
        if self.autopilot_enabled:
            return
        keys = pygame.key.get_pressed()

        self.control.throttle = 1.0 if keys[pygame.K_UP] else 0.0
        self.control.brake = 1.0 if keys[pygame.K_DOWN] else 0.0

        steer = 0.0
        if keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_RIGHT]:
            steer += 1.0
        self.control.steer = steer

        # Correct CARLA reverse handling
        self.control.manual_gear_shift = True
        self.control.reverse = self.in_reverse
        self.control.gear = -1 if self.in_reverse else 1

        self.vehicle.apply_control(self.control)

    def _cycle_map(self):
        logger.info("Cycling map...")

        self.cleanup_actors()

        self.map_index = (self.map_index + 1) % len(self.available_maps)
        next_map = self.available_maps[self.map_index]

        self._load_world(next_map)
        self._spawn_vehicle()
        self._spawn_background_agents()
        self._setup_camera()
        self._setup_data_collector()

        logger.info(f"[MAP] Switched to {next_map}")

    def _cycle_map_delta(self, delta):
        if not self.available_maps:
            return
        self.map_index = (self.map_index + delta) % len(self.available_maps)
        next_map = self.available_maps[self.map_index]

        logger.info(f"Cycling map ({delta:+d})...")
        self.cleanup_actors()
        self._load_world(next_map)
        self._spawn_vehicle()
        self._spawn_background_agents()
        self._setup_camera()
        self._setup_data_collector()
        logger.info(f"[MAP] Switched to {next_map}")

    def _teleport_next_spawn(self):
        if not self.vehicle:
            return
        self.spawn_index = (self.spawn_index + 1) % len(self.spawn_points)
        transform = self.spawn_points[self.spawn_index]
        self.vehicle.set_transform(transform)
        self._reset_vehicle_motion()
        logger.info(f"[SPAWN] Teleport to idx={self.spawn_index}")

    def _teleport_spawn_delta(self, delta):
        if not self.vehicle:
            return
        self.spawn_index = (self.spawn_index + delta) % len(self.spawn_points)
        transform = self.spawn_points[self.spawn_index]
        self.vehicle.set_transform(transform)
        self._reset_vehicle_motion()
        logger.info(f"[SPAWN] Teleport to idx={self.spawn_index}")

    def _log_controls(self):
        logger.info(
            "Controls: Arrows=Drive | Y=Toggle Reverse | P=Toggle Autopilot | "
            "I=Next Spawn U=Prev Spawn | M=Next Map N=Prev Map | R=Record | "
            "B=Toggle Minimap | K=Inference | G=DAgger | J=RL | ESC=Quit"
        )

    # ============================
    # Sensors
    # ============================
    def _on_image(self, image):
        self.latest_image = image.raw_data
        self.latest_image_dims = (image.width, image.height)

    # ============================
    # Main Loop
    # ============================
    def run(self):
        try:
            while self.running:
                self.frame_count += 1
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False

                        if event.key == pygame.K_y:
                            self.in_reverse = not self.in_reverse
                            logger.info(
                                f"[GEAR] {'REVERSE' if self.in_reverse else 'FORWARD'}"
                            )
                        if event.key == pygame.K_p:
                            if self.dagger_enabled:
                                self._set_dagger_enabled(False)
                            if self.rl_enabled:
                                self._set_reinforcement_enabled(False)
                            self.autopilot_enabled = not self.autopilot_enabled
                            if self.autopilot_enabled:
                                self._set_inference_enabled(False)
                            if self.vehicle:
                                if self.autopilot_enabled:
                                    self.control = carla.VehicleControl()
                                    self.in_reverse = False
                                if self.tm:
                                    self.vehicle.set_autopilot(
                                        self.autopilot_enabled, self.tm.get_port()
                                    )
                                else:
                                    self.vehicle.set_autopilot(self.autopilot_enabled)
                            logger.info(
                                f"[AUTOPILOT] {'ENABLED' if self.autopilot_enabled else 'DISABLED'}"
                            )

                        if event.key == pygame.K_k:
                            self._set_inference_enabled(not self.inference_enabled)
                            if self.inference_enabled:
                                self._set_dagger_enabled(False)
                                self._set_reinforcement_enabled(False)

                        if event.key == pygame.K_g:
                            self._set_dagger_enabled(not self.dagger_enabled)
                            if self.dagger_enabled:
                                self._set_inference_enabled(False)
                                self._set_reinforcement_enabled(False)

                        if event.key == pygame.K_j:
                            self._set_reinforcement_enabled(not self.rl_enabled)
                            if self.rl_enabled:
                                self._set_inference_enabled(False)
                                self._set_dagger_enabled(False)

                        if event.key == pygame.K_r:
                            if self.recorder.recording:
                                self.stop_recording()
                            else:
                                self.start_recording()

                        if event.key == pygame.K_b:
                            self.minimap_enabled = not self.minimap_enabled
                            if self.minimap_enabled and self.minimap_surface is None:
                                self._build_minimap()
                            logger.info(
                                f"[MINIMAP] {'ENABLED' if self.minimap_enabled else 'DISABLED'}"
                            )

                        if event.key == pygame.K_m:
                            self._cycle_map_delta(1)
                        if event.key == pygame.K_n:
                            self._cycle_map_delta(-1)
                        if event.key == pygame.K_i:
                            self._teleport_spawn_delta(1)
                        if event.key == pygame.K_u:
                            self._teleport_spawn_delta(-1)

                if self.dagger_enabled:
                    self._apply_dagger()
                elif self.rl_enabled:
                    self._apply_reinforcement()
                elif self.inference_enabled:
                    self._apply_inference()
                else:
                    self._handle_controls()

                if self.world is not None:
                    try:
                        if self.cfg.get("synchronous_mode", False):
                            self.world.tick()
                        else:
                            self.world.wait_for_tick()
                    except Exception:
                        logger.exception("[CARLA] world frame advance failed")

                if (self.autopilot_enabled or self.inference_enabled) and self.frame_count % 25 == 0:
                    self._future_waypoint_path(step_m=2.0)
                    self._debug_waypoints()

                if self.recorder.recording and self.data_collector is not None and not self.dagger_enabled:
                    sample = self.data_collector.collect_step()
                    if sample is not None:
                        self.recorder.append(*sample)

                if self.latest_image is not None and self.latest_image_dims is not None:
                    width, height = self.latest_image_dims
                    frame = np.frombuffer(self.latest_image, dtype=np.uint8)
                    frame = frame.reshape((height, width, 4))
                    frame = frame[:, :, :3]
                    self.image_surface = pygame.surfarray.make_surface(
                        frame.swapaxes(0, 1)[:, :, ::-1]
                    )
                    self.latest_image = None

                if self.image_surface is not None:
                    self.display.blit(self.image_surface, (0, 0))
                self._draw_minimap()
                pygame.display.update()

                self.clock.tick_busy_loop(60)

        finally:
            self.cleanup()

    def start_recording(self):
        if self.world is None or self.vehicle is None:
            return
        if self.data_collector is None:
            self._setup_data_collector()
        if self.data_collector is not None:
            try:
                self.data_collector.reset_alignment_stats()
            except Exception:
                logger.exception("[RECORD] Failed to reset alignment stats")
        map_name = self.world.get_map().name if self.world else "unknown"
        self.recorder.start(map_name, self.spawn_index)

    def stop_recording(self):
        alignment_stats = {}
        if self.data_collector is not None:
            try:
                alignment_stats = self.data_collector.get_alignment_stats()
            except Exception:
                logger.exception("[RECORD] Failed to collect alignment stats")
        summary = self.recorder.stop(alignment_stats=alignment_stats)
        if summary:
            logger.info(f"[RECORD] Done: {summary['record_path']}")

    def _reset_vehicle_motion(self):
        if not self.vehicle:
            return
        # Clear velocity + inputs after teleport to avoid carrying momentum.
        try:
            self.vehicle.set_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.vehicle.set_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:
            pass
        self.control = carla.VehicleControl()
        self.control.manual_gear_shift = True
        self.control.reverse = False
        self.control.gear = 1
        self.vehicle.apply_control(self.control)
        self._vision_history.clear()
        self._lidar_history.clear()
        self._telemetry_history.clear()
        self._waypoint_history.clear()
        if self.data_collector is not None:
            try:
                self.data_collector.clear_collision()
                self.data_collector.chain_count = 0
                self.data_collector.prev_steer = 0.0
            except Exception:
                logger.exception("[CONTROL] Failed to reset collector state after teleport")

    def cleanup(self):
        logger.info("Cleaning up...")
        if self.recorder.recording:
            self.stop_recording()
        try:
            if self.dagger_recorder.recording:
                alignment_stats = (
                    self.data_collector.get_alignment_stats()
                    if self.data_collector is not None
                    else {}
                )
                self.dagger_recorder.stop(alignment_stats=alignment_stats)
        except Exception:
            logger.exception("[DAGGER] Failed to stop recorder on cleanup")
        self._flush_reinforcement(finalize=False)
        self.cleanup_actors()
        pygame.quit()

    def _load_inference_model(self):
        if self.inference_model is not None:
            return True
        if (
            not str(self.inference_model_path).startswith("s3://")
            and not os.path.exists(self.inference_model_path)
        ):
            logger.error(f"[INFER] Model not found: {self.inference_model_path}")
            return False
        self.inference_model = self.inference_trainer.load(self.inference_model_path)
        logger.info(f"[INFER] Loaded control model: {self.inference_model_path}")
        return True

    def _set_inference_enabled(self, enabled):
        self.inference_enabled = bool(enabled)
        if self.inference_enabled:
            if not self._load_inference_model():
                self.inference_enabled = False
                return
            self._vision_history.clear()
            self._lidar_history.clear()
            self._telemetry_history.clear()
            self._waypoint_history.clear()
            self._infer_log_step = 0
            if self.vehicle:
                self.autopilot_enabled = False
                self.control = carla.VehicleControl()
                self.in_reverse = False
                if self.tm:
                    self.vehicle.set_autopilot(False, self.tm.get_port())
                else:
                    self.vehicle.set_autopilot(False)
            logger.info("[INFER] ENABLED")
        else:
            logger.info("[INFER] DISABLED")

    def _set_autopilot(self, enabled):
        self.autopilot_enabled = bool(enabled)
        if not self.vehicle:
            return
        if self.autopilot_enabled:
            self.control = carla.VehicleControl()
            self.in_reverse = False
        if self.tm:
            self.vehicle.set_autopilot(self.autopilot_enabled, self.tm.get_port())
        else:
            self.vehicle.set_autopilot(self.autopilot_enabled)

    def _set_dagger_enabled(self, enabled):
        self.dagger_enabled = bool(enabled)
        if self.dagger_enabled:
            if not self._load_inference_model():
                self.dagger_enabled = False
                return
            self.dagger_steps = 0
            self.dagger_batch = []
            self._vision_history.clear()
            self._lidar_history.clear()
            self._telemetry_history.clear()
            self._waypoint_history.clear()
            self._infer_log_step = 0
            self.dagger_controller = "model"
            self.dagger_window_step = 0
            self._set_autopilot(False)
            logger.info(
                f"[DAGGER] ENABLED window={self.dagger_window} "
                f"infer={self.dagger_infer_steps} autopilot={self.dagger_autopilot_steps} "
                f"checkpoint_every={self.dagger_checkpoint_every}"
            )
        else:
            self._set_autopilot(False)
            try:
                if self.dagger_recorder.recording:
                    alignment_stats = (
                        self.data_collector.get_alignment_stats()
                        if self.data_collector is not None
                        else {}
                    )
                    self.dagger_recorder.stop(alignment_stats=alignment_stats)
            except Exception:
                logger.exception("[DAGGER] Failed to stop recorder")
            self.dagger_controller = "model"
            logger.info("[DAGGER] DISABLED")

    def _set_reinforcement_enabled(self, enabled):
        self.rl_enabled = bool(enabled)
        if self.rl_enabled:
            if not self._load_inference_model():
                self.rl_enabled = False
                return
            self.rl_steps = 0
            self.rl_rollout = []
            self.rl_pending = None
            self._vision_history.clear()
            self._lidar_history.clear()
            self._telemetry_history.clear()
            self._waypoint_history.clear()
            self._infer_log_step = 0
            self.rl_collision_cooldown = 10
            if self.data_collector is not None:
                try:
                    self.data_collector.clear_collision()
                except Exception:
                    logger.exception("[RL] Failed to clear collision state on enable")
            self._set_autopilot(False)
            logger.info(
                "[RL] ENABLED rollout=%d gamma=%.3f noise=%s checkpoint_every=%d",
                self.rl_rollout_steps,
                self.rl_gamma,
                np.array2string(self.rl_action_noise, precision=3),
                self.rl_checkpoint_every,
            )
        else:
            self._flush_reinforcement(finalize=True)
            self._set_autopilot(False)
            logger.info("[RL] DISABLED")

    @staticmethod
    def _controls_to_vehicle_control(controls):
        throttle = float(np.clip(controls[0], 0.0, 1.0))
        brake = float(np.clip(controls[1], 0.0, 1.0))
        steer = float(controls[2])
        steer = float(np.clip(steer, -1.0, 1.0))
        return throttle, brake, steer

    def _sample_rl_action(self, pred_ctrl):
        noise = np.random.normal(loc=0.0, scale=self.rl_action_noise, size=3).astype(np.float32)
        action = np.asarray(pred_ctrl, dtype=np.float32) + noise
        throttle, brake, steer = self._controls_to_vehicle_control(action)
        return np.asarray([throttle, brake, steer], dtype=np.float32)

    def _train_reinforcement_rollout(self):
        if not self.rl_rollout or self.inference_model is None:
            return

        rewards = np.asarray([step["reward"] for step in self.rl_rollout], dtype=np.float32)
        returns = np.zeros_like(rewards)
        running = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            running = float(rewards[idx]) + (self.rl_gamma * running)
            returns[idx] = running

        mean_return = float(np.mean(returns))
        if self.rl_reward_ema is None:
            self.rl_reward_ema = mean_return
        else:
            a = self.rl_reward_alpha
            self.rl_reward_ema = (a * self.rl_reward_ema) + ((1.0 - a) * mean_return)

        advantages = (returns - float(self.rl_reward_ema)) / self.rl_reward_temperature
        weights = np.exp(np.clip(advantages, -4.0, 4.0)).astype(np.float32)

        v_b = tf.convert_to_tensor(np.stack([s["vision"] for s in self.rl_rollout], axis=0), dtype=tf.float32)
        l_b = tf.convert_to_tensor(np.stack([s["lidar"] for s in self.rl_rollout], axis=0), dtype=tf.float32)
        t_b = tf.convert_to_tensor(np.stack([s["telemetry"] for s in self.rl_rollout], axis=0), dtype=tf.float32)
        w_b = tf.convert_to_tensor(np.stack([s["waypoints"] for s in self.rl_rollout], axis=0), dtype=tf.float32)
        y_b = tf.convert_to_tensor(np.stack([s["action"] for s in self.rl_rollout], axis=0), dtype=tf.float32)
        weights_b = tf.convert_to_tensor(weights[:, None], dtype=tf.float32)

        with tf.GradientTape() as tape:
            pred_ctrl_b = self.inference_trainer._control_output(
                self.inference_model([v_b, l_b, t_b, w_b], training=True)
            )
            sq_err = tf.square(pred_ctrl_b - y_b)
            weighted_loss = tf.reduce_mean(tf.reduce_mean(sq_err * weights_b, axis=1))
        grads = tape.gradient(weighted_loss, self.inference_model.trainable_variables)
        grad_vars = [(g, v) for g, v in zip(grads, self.inference_model.trainable_variables) if g is not None]
        if not grad_vars:
            logger.warning("[RL] No gradients produced for rollout update")
            self.rl_rollout = []
            return
        self.rl_optimizer.apply_gradients(grad_vars)

        if self.rl_reward_log_every > 0 and self.rl_steps % self.rl_reward_log_every == 0:
            logger.info(
                "[RL] step=%d loss=%.5f reward_mean=%.5f return_mean=%.5f reward_ema=%.5f",
                self.rl_steps,
                float(weighted_loss),
                float(np.mean(rewards)),
                mean_return,
                0.0 if self.rl_reward_ema is None else float(self.rl_reward_ema),
            )

        self.rl_rollout = []

    def _flush_reinforcement(self, finalize):
        if self.rl_pending is not None:
            pending = dict(self.rl_pending)
            pending["reward"] = 0.0
            self.rl_rollout.append(pending)
            self.rl_pending = None
        if self.rl_rollout:
            self._train_reinforcement_rollout()
        if finalize and self.rl_checkpoint_every > 0 and self.rl_steps > 0:
            try:
                self.inference_trainer.save(f"rl_step_{self.rl_steps:07d}.keras")
            except Exception:
                logger.exception("[RL] Failed to save checkpoint")

    def _apply_reinforcement(self):
        if self.data_collector is None or self.inference_model is None:
            return
        try:
            sample = self.data_collector.collect_step(include_reward=True)
            if sample is None:
                return
            vision, lidar, telemetry, waypoints, _, reward, _ = sample
            if self.rl_pending is not None:
                pending = dict(self.rl_pending)
                pending["reward"] = float(reward)
                self.rl_rollout.append(pending)
                self.rl_pending = None

            collision = 0.0
            if self.rl_collision_cooldown > 0:
                self.rl_collision_cooldown -= 1
                try:
                    self.data_collector.clear_collision()
                except Exception:
                    logger.exception("[RL] Failed to clear collision during cooldown")
            else:
                collision = self.data_collector.pop_collision()

            if collision > 0.0:
                logger.info("[RL] Collision detected (%.2f) -> train/teleport", collision)
                self._flush_reinforcement(finalize=False)
                self.rl_collision_cooldown = 10
                self._teleport_next_spawn()
                return

            vision, lidar, telemetry, waypoints = self._preprocess_model_inputs(
                vision, lidar, telemetry, waypoints
            )
            self._vision_history.append(vision)
            self._lidar_history.append(lidar)
            self._telemetry_history.append(telemetry)
            self._waypoint_history.append(waypoints)

            vision_b = self._stack_history(self._vision_history, vision)
            lidar_b = self._stack_history(self._lidar_history, lidar)
            telemetry_b = self._stack_history(self._telemetry_history, telemetry)
            waypoints_b = self._stack_history(self._waypoint_history, waypoints)

            pred_ctrl = self.inference_trainer._control_output(self.inference_model(
                [
                    tf.convert_to_tensor(vision_b[None, ...], dtype=tf.float32),
                    tf.convert_to_tensor(lidar_b[None, ...], dtype=tf.float32),
                    tf.convert_to_tensor(telemetry_b[None, ...], dtype=tf.float32),
                    tf.convert_to_tensor(waypoints_b[None, ...], dtype=tf.float32),
                ],
                training=False,
            ))
            if hasattr(pred_ctrl, "numpy"):
                pred_ctrl = pred_ctrl.numpy()
            raw_ctrl = np.asarray(pred_ctrl[0], dtype=np.float32)
            sampled_ctrl = self._sample_rl_action(raw_ctrl)
            throttle, brake, steer = self._controls_to_vehicle_control(sampled_ctrl)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            )
            logger.info(
                "[RL] step=%d reward=%.4f raw=%.3f,%.3f,%.3f act=%.3f,%.3f,%.3f",
                self.rl_steps,
                float(reward),
                float(raw_ctrl[0]),
                float(raw_ctrl[1]),
                float(raw_ctrl[2]),
                throttle,
                brake,
                steer,
            )
            self.rl_pending = {
                "vision": np.asarray(vision_b, dtype=np.float32),
                "lidar": np.asarray(lidar_b, dtype=np.float32),
                "telemetry": np.asarray(telemetry_b, dtype=np.float32),
                "waypoints": np.asarray(waypoints_b, dtype=np.float32),
                "action": np.asarray([throttle, brake, steer], dtype=np.float32),
            }
            self.rl_steps += 1

            if self.rl_rollout_steps > 0 and len(self.rl_rollout) >= self.rl_rollout_steps:
                self._train_reinforcement_rollout()

            if self.rl_checkpoint_every > 0 and self.rl_steps % self.rl_checkpoint_every == 0:
                try:
                    self.inference_trainer.save(f"rl_step_{self.rl_steps:07d}.keras")
                    logger.info("[RL] Saved checkpoint at step %d", self.rl_steps)
                except Exception:
                    logger.exception("[RL] Failed to save checkpoint")
        except Exception:
            logger.exception("[RL] _apply_reinforcement failed")

    def _apply_dagger(self):
        if self.data_collector is None or self.inference_model is None:
            return
        window_size = max(1, self.dagger_window)
        window_step = self.dagger_window_step
        window_idx = self.dagger_steps // window_size
        controller = "expert" if window_step >= self.dagger_infer_steps else "model"
        if controller != self.dagger_controller:
            if self.dagger_controller == "expert" and self.dagger_recorder.recording:
                try:
                    alignment_stats = (
                        self.data_collector.get_alignment_stats()
                        if self.data_collector is not None
                        else {}
                    )
                    self.dagger_recorder.stop(alignment_stats=alignment_stats)
                except Exception:
                    logger.exception("[DAGGER] Failed to stop recorder")
            if controller == "expert":
                try:
                    if self.data_collector is not None:
                        self.data_collector.reset_alignment_stats()
                    self.dagger_recorder.start(
                        self.world.get_map().name if self.world else "unknown",
                        self.spawn_index,
                    )
                except Exception:
                    logger.exception("[DAGGER] Failed to start recorder")
            self.dagger_controller = controller
        self._set_autopilot(controller == "expert")
        logger.info(
            f"[DAGGER] step={self.dagger_steps} window={window_idx} "
            f"window_step={window_step}/{window_size} controller={controller}"
        )
        if controller == "model":
            self._apply_inference()
        else:
            sample = self.data_collector.collect_step()
            if sample is not None:
                vision, lidar, telemetry, waypoints, expert_ctrl = sample
                if self.dagger_recorder.recording:
                    self.dagger_recorder.append(vision, lidar, telemetry, waypoints, expert_ctrl)
                vision, lidar, telemetry, waypoints = self._preprocess_model_inputs(
                    vision, lidar, telemetry, waypoints
                )
                self._vision_history.append(vision)
                self._lidar_history.append(lidar)
                self._telemetry_history.append(telemetry)
                self._waypoint_history.append(waypoints)
                self.dagger_batch.append(
                    (
                        self._stack_history(self._vision_history, vision),
                        self._stack_history(self._lidar_history, lidar),
                        self._stack_history(self._telemetry_history, telemetry),
                        self._stack_history(self._waypoint_history, waypoints),
                        np.asarray(expert_ctrl, dtype=np.float32),
                    )
                )
                if len(self.dagger_batch) >= BATCH_SIZE:
                    v_b, l_b, t_b, w_b, y_b = zip(*self.dagger_batch[:BATCH_SIZE])
                    self.dagger_batch = self.dagger_batch[BATCH_SIZE:]
                    v_b = tf.convert_to_tensor(np.stack(v_b, axis=0), dtype=tf.float32)
                    l_b = tf.convert_to_tensor(np.stack(l_b, axis=0), dtype=tf.float32)
                    t_b = tf.convert_to_tensor(np.stack(t_b, axis=0), dtype=tf.float32)
                    w_b = tf.convert_to_tensor(np.stack(w_b, axis=0), dtype=tf.float32)
                    y_b = tf.convert_to_tensor(np.stack(y_b, axis=0), dtype=tf.float32)
                    with tf.GradientTape() as tape:
                        pred_ctrl_b = self.inference_trainer._control_output(
                            self.inference_model([v_b, l_b, t_b, w_b], training=True)
                        )
                        batch_loss = tf.reduce_mean(tf.square(pred_ctrl_b - y_b))
                    grads = tape.gradient(batch_loss, self.inference_model.trainable_variables)
                    grad_vars = [
                        (g, v)
                        for g, v in zip(grads, self.inference_model.trainable_variables)
                        if g is not None
                    ]
                    self.dagger_optimizer.apply_gradients(grad_vars)
                    loss = float(batch_loss)
                    if self.dagger_loss_ema is None:
                        self.dagger_loss_ema = loss
                    else:
                        a = self.dagger_loss_alpha
                        self.dagger_loss_ema = a * self.dagger_loss_ema + (1.0 - a) * loss
                    if self.dagger_loss_log_every > 0 and self.dagger_steps % self.dagger_loss_log_every == 0:
                        logger.info(
                            f"[DAGGER] Loss step={self.dagger_steps} "
                            f"loss={loss:.5f} ema={self.dagger_loss_ema:.5f}"
                        )
        self.dagger_steps += 1
        self.dagger_window_step = (self.dagger_window_step + 1) % window_size
        if self.dagger_checkpoint_every > 0 and self.dagger_steps % self.dagger_checkpoint_every == 0:
            try:
                self.inference_trainer.save(f"dagger_step_{self.dagger_steps:07d}.keras")
                if self.dagger_loss_ema is None:
                    logger.info(f"[DAGGER] Saved checkpoint at step {self.dagger_steps}")
                else:
                    logger.info(
                        f"[DAGGER] Saved checkpoint at step {self.dagger_steps} "
                        f"loss_ema={self.dagger_loss_ema:.5f}"
                    )
            except Exception:
                logger.exception("[DAGGER] Failed to save checkpoint")
        collision = self.data_collector.pop_collision()
        if collision > 0.0:
            logger.info(f"[DAGGER] Collision detected ({collision:.2f}) -> teleport")
            self._teleport_next_spawn()

    def _apply_inference(self):
        if self.data_collector is None or self.inference_model is None:
            return
        try:
            sample = self.data_collector.collect_step()
            if sample is None:
                return
            vision, lidar, telemetry, waypoints, _ = sample
            vision, lidar, telemetry, waypoints = self._preprocess_model_inputs(
                vision, lidar, telemetry, waypoints
            )
            self._vision_history.append(vision)
            self._lidar_history.append(lidar)
            self._telemetry_history.append(telemetry)
            self._waypoint_history.append(waypoints)
            vision = tf.convert_to_tensor(self._stack_history(self._vision_history, vision)[None, ...], dtype=tf.float32)
            lidar = tf.convert_to_tensor(self._stack_history(self._lidar_history, lidar)[None, ...], dtype=tf.float32)
            telemetry = tf.convert_to_tensor(self._stack_history(self._telemetry_history, telemetry)[None, ...], dtype=tf.float32)
            waypoints = tf.convert_to_tensor(self._stack_history(self._waypoint_history, waypoints)[None, ...], dtype=tf.float32)

            pred_ctrl = self.inference_trainer._control_output(
                self.inference_model([vision, lidar, telemetry, waypoints], training=False)
            )
            if hasattr(pred_ctrl, "numpy"):
                pred_ctrl = pred_ctrl.numpy()
            ctrl = np.asarray(pred_ctrl[0], dtype=np.float32)
            raw_throttle = float(ctrl[0])
            raw_brake = float(ctrl[1])
            raw_steer = float(ctrl[2])
            throttle, brake, steer = self._controls_to_vehicle_control(ctrl)
            speed_mps = float(self._telemetry_history[-1][0]) if self._telemetry_history else 0.0
            speed_kph = float(self._telemetry_history[-1][1]) if self._telemetry_history else 0.0
            self._infer_log_step += 1
            if self._infer_log_step <= 5 or self._infer_log_step % INFER_LOG_EVERY == 0:
                logger.info(
                    "[INFER] step=%d hist=%d speed_mps=%.3f speed_kph=%.3f "
                    "raw=%.4f,%.4f,%.4f clipped=%.4f,%.4f,%.4f "
                    "applied=%.4f,%.4f,%.4f",
                    self._infer_log_step,
                    len(self._telemetry_history),
                    speed_mps,
                    speed_kph,
                    raw_throttle,
                    raw_brake,
                    raw_steer,
                    float(np.clip(raw_throttle, 0.0, 1.0)),
                    float(np.clip(raw_brake, 0.0, 1.0)),
                    float(np.clip(raw_steer, -1.0, 1.0)),
                    throttle,
                    brake,
                    steer,
                )
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            )
        except Exception:
            logger.exception("[INFER] _apply_inference failed")


# ============================
# Entry Point
# ============================
if __name__ == "__main__":
    # python drive.py [MAP] [POLICY_PATH]   e.g. python drive.py Town02 artifacts/policy.keras
    CONFIG["map"] = "Town01"
    if len(sys.argv) > 1:
        CONFIG["map"] = sys.argv[1]
    if len(sys.argv) > 2:
        CONFIG["inference_model_path"] = sys.argv[2]
    app = CarlaCameraApp(CONFIG)
    app.initialize()
    app.run()
