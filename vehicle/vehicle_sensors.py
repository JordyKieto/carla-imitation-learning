import collections
import math
import threading
import logging
import numpy as np
import carla

from utils.config import CONFIG

# ============================================================
# SHORT CONFIG ALIASES
# ============================================================
CAM_W = int(CONFIG["sensors"]["camera"]["width"])
CAM_H = int(CONFIG["sensors"]["camera"]["height"])

LIDAR_H = int(CONFIG["sensors"]["lidar"]["height"])
LIDAR_W = int(CONFIG["sensors"]["lidar"]["width"])

HIST_X_MIN = float(CONFIG["sensors"]["lidar"]["hist"]["x_min"])
HIST_X_MAX = float(CONFIG["sensors"]["lidar"]["hist"]["x_max"])
HIST_Y_MIN = float(CONFIG["sensors"]["lidar"]["hist"]["y_min"])
HIST_Y_MAX = float(CONFIG["sensors"]["lidar"]["hist"]["y_max"])
LIDAR_PPS = CONFIG["sensors"]["lidar"]["points_per_second"]
FIXED_DELTA_SECONDS = float(CONFIG["carla"]["fixed_delta_seconds"])
PAIR_MAX_FRAME_GAP = 1
PAIR_MAX_TS_DELTA = FIXED_DELTA_SECONDS * 1.5
PAIR_BUFFER_SIZE = 8
ALIGNMENT_LOG_EVERY = 50

logger = logging.getLogger("VehicleSensors")

class VehicleSensors:
    """
    Optimized CARLA sensor rig.
    FIXED: BGRA -> RGB conversion for pretrained model compatibility.
    FIXED: Native resolution capture (224x224) to eliminate resizing artifacts.
    """

    def __init__(
        self,
        *args,
        enable_camera: bool = True,
        enable_semantic: bool = True,
        enable_lidar: bool = True,
        enable_collision: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        print(f"[INIT] Vehicle Sensors | Native Resolution: {CAM_W}x{CAM_H}")
        self._lock = threading.Lock()
        self._alive = True

        # Preallocated RGB buffer (3 channels, not 4)
        self._latest_img_u8 = np.empty((CAM_H, CAM_W, 3), dtype=np.uint8)
        self._has_img = False
        self._latest_img_frame = -1
        self._latest_img_ts = None
        self._camera_buffer = collections.deque(maxlen=PAIR_BUFFER_SIZE)
        self._latest_semantic_u8 = np.empty((CAM_H, CAM_W, 3), dtype=np.uint8)
        self._has_semantic = False

        self._latest_lidar_f32 = np.zeros((LIDAR_H, LIDAR_W), dtype=np.float32)
        self._has_lidar = False
        self._latest_lidar_frame = -1
        self._latest_lidar_ts = None
        self._lidar_buffer = collections.deque(maxlen=PAIR_BUFFER_SIZE)
        self._latest_collision = 0.0

        self.camera = None
        self.semantic_camera = None
        self.lidar = None
        self.collision_sensor = None

        self._lidar_max_points = max(2048, int(LIDAR_PPS * 1.5))
        self._latest_lidar_pts = np.empty((self._lidar_max_points, 4), dtype=np.float32)
        self._latest_lidar_n = 0
        self._lidar_version = 0  
        self._hist_version = -1
        self._reset_alignment_stats()

        if enable_camera:
            self._attach_camera()
            if enable_semantic:
                self._attach_semantic_camera()
        if enable_lidar:
            self._attach_lidar()
        if enable_collision:
            self._attach_collision_sensor()

    def ready(self) -> bool:
        with self._lock:
            has_semantic = (self.semantic_camera is None) or self._has_semantic
            has_lidar = (self.lidar is None) or self._has_lidar
            return bool(self._has_img and has_lidar and has_semantic)

    def _reset_alignment_stats(self):
        self._alignment_stats = {
            "accepted": 0,
            "accepted_exact": 0,
            "accepted_frame_gap_1": 0,
            "accepted_frame_gap_2": 0,
            "accepted_frame_gap_sum": 0.0,
            "accepted_ts_delta_sum": 0.0,
            "rejected_missing": 0,
            "rejected_frame_gap": 0,
            "rejected_time_gap": 0,
            "last_accept": None,
            "last_reject": None,
        }

    def reset_alignment_stats(self):
        with self._lock:
            self._reset_alignment_stats()

    def get_alignment_stats(self):
        with self._lock:
            stats = dict(self._alignment_stats)
        total_rejected = (
            stats["rejected_missing"]
            + stats["rejected_frame_gap"]
            + stats["rejected_time_gap"]
        )
        total_seen = stats["accepted"] + total_rejected
        stats["total_rejected"] = total_rejected
        stats["total_seen"] = total_seen
        stats["accept_rate"] = (stats["accepted"] / total_seen) if total_seen > 0 else 0.0
        accepted = stats["accepted"]
        stats["accepted_mean_frame_gap"] = (
            stats["accepted_frame_gap_sum"] / accepted if accepted > 0 else 0.0
        )
        stats["accepted_mean_timestamp_delta_sec"] = (
            stats["accepted_ts_delta_sum"] / accepted if accepted > 0 else 0.0
        )
        stats["policy"] = {
            "max_frame_gap": PAIR_MAX_FRAME_GAP,
            "max_timestamp_delta_sec": PAIR_MAX_TS_DELTA,
            "buffer_size": PAIR_BUFFER_SIZE,
        }
        return stats

    def _record_alignment_reject(self, reason, img_frame, lidar_frame, ts_delta):
        self._alignment_stats[reason] += 1
        self._alignment_stats["last_reject"] = {
            "reason": reason,
            "camera_frame": None if img_frame is None else int(img_frame),
            "lidar_frame": None if lidar_frame is None else int(lidar_frame),
            "frame_gap": None if (img_frame is None or lidar_frame is None) else int(abs(img_frame - lidar_frame)),
            "timestamp_delta_sec": None if ts_delta is None else float(ts_delta),
        }
        total_rejected = (
            self._alignment_stats["rejected_missing"]
            + self._alignment_stats["rejected_frame_gap"]
            + self._alignment_stats["rejected_time_gap"]
        )
        if total_rejected <= 5 or total_rejected % ALIGNMENT_LOG_EVERY == 0:
            rej = self._alignment_stats["last_reject"]
            logger.info(
                "[ALIGN] reject=%s cam=%s lidar=%s frame_gap=%s ts_delta=%.6f accepted=%d rejected=%d",
                rej["reason"],
                rej["camera_frame"],
                rej["lidar_frame"],
                rej["frame_gap"],
                0.0 if rej["timestamp_delta_sec"] is None else rej["timestamp_delta_sec"],
                self._alignment_stats["accepted"],
                total_rejected,
            )

    def _record_alignment_accept(self, img_frame, lidar_frame, ts_delta):
        frame_gap = int(abs(img_frame - lidar_frame))
        self._alignment_stats["accepted"] += 1
        if frame_gap == 0:
            self._alignment_stats["accepted_exact"] += 1
        elif frame_gap == 1:
            self._alignment_stats["accepted_frame_gap_1"] += 1
        elif frame_gap == 2:
            self._alignment_stats["accepted_frame_gap_2"] += 1
        self._alignment_stats["accepted_frame_gap_sum"] += float(frame_gap)
        self._alignment_stats["accepted_ts_delta_sum"] += float(ts_delta)
        self._alignment_stats["last_accept"] = {
            "camera_frame": int(img_frame),
            "lidar_frame": int(lidar_frame),
            "frame_gap": frame_gap,
            "timestamp_delta_sec": float(ts_delta),
        }
        accepted = self._alignment_stats["accepted"]
        if accepted <= 5 or accepted % ALIGNMENT_LOG_EVERY == 0:
            mean_frame_gap = self._alignment_stats["accepted_frame_gap_sum"] / accepted
            mean_ts_delta = self._alignment_stats["accepted_ts_delta_sum"] / accepted
            logger.info(
                "[ALIGN] accept cam=%d lidar=%d frame_gap=%d ts_delta=%.6f exact=%d gap1=%d mean_frame_gap=%.3f mean_ts_delta=%.6f",
                int(img_frame),
                int(lidar_frame),
                frame_gap,
                float(ts_delta),
                self._alignment_stats["accepted_exact"],
                self._alignment_stats["accepted_frame_gap_1"],
                mean_frame_gap,
                mean_ts_delta,
            )

    def _select_paired_samples_locked(self):
        if not self._camera_buffer or not self._lidar_buffer:
            return None

        best = None
        for cam_idx, cam_sample in enumerate(self._camera_buffer):
            cam_frame = int(cam_sample["frame"])
            cam_ts = float(cam_sample["timestamp"])
            for lidar_idx, lidar_sample in enumerate(self._lidar_buffer):
                lidar_frame = int(lidar_sample["frame"])
                lidar_ts = float(lidar_sample["timestamp"])
                frame_gap = abs(cam_frame - lidar_frame)
                ts_delta = abs(cam_ts - lidar_ts)
                if frame_gap > PAIR_MAX_FRAME_GAP or ts_delta > PAIR_MAX_TS_DELTA:
                    continue
                freshness = max(cam_frame, lidar_frame)
                rank = (frame_gap, ts_delta, -freshness)
                if best is None or rank < best["rank"]:
                    best = {
                        "rank": rank,
                        "cam_idx": cam_idx,
                        "lidar_idx": lidar_idx,
                        "camera": cam_sample,
                        "lidar": lidar_sample,
                    }
        if best is None:
            return None

        cam_idx = best["cam_idx"]
        lidar_idx = best["lidar_idx"]
        for _ in range(cam_idx + 1):
            self._camera_buffer.popleft()
        for _ in range(lidar_idx + 1):
            self._lidar_buffer.popleft()
        return best["camera"], best["lidar"]

    def snapshot(self):
        with self._lock:
            if not (self._has_img and ((self.lidar is None) or self._has_lidar)):
                self._record_alignment_reject("rejected_missing", self._latest_img_frame, self._latest_lidar_frame, None)
                return None

            semantic = self._latest_semantic_u8.copy() if self._has_semantic else None
            collision = float(self._latest_collision)
            if self.lidar is None:
                img = self._latest_img_u8.copy()
                img_frame = int(self._latest_img_frame)
                img_ts = self._latest_img_ts
                self._record_alignment_accept(img_frame, img_frame, 0.0)
                return (img, semantic, self._latest_lidar_f32.copy(), collision, img_frame, img_ts)

            if not self._camera_buffer or not self._lidar_buffer:
                self._record_alignment_reject("rejected_missing", self._latest_img_frame, self._latest_lidar_frame, None)
                return None

            paired = self._select_paired_samples_locked()
            if paired is None:
                img_frame = int(self._latest_img_frame)
                lidar_frame = int(self._latest_lidar_frame)
                img_ts = self._latest_img_ts
                lidar_ts = self._latest_lidar_ts
                ts_delta = None if img_ts is None or lidar_ts is None else abs(float(img_ts) - float(lidar_ts))
                frame_gap = abs(img_frame - lidar_frame) if img_frame >= 0 and lidar_frame >= 0 else None
                if frame_gap is not None and frame_gap > PAIR_MAX_FRAME_GAP:
                    self._record_alignment_reject("rejected_frame_gap", img_frame, lidar_frame, ts_delta)
                else:
                    self._record_alignment_reject("rejected_time_gap", img_frame, lidar_frame, ts_delta)
                return None

            cam_sample, lidar_sample = paired
            img = cam_sample["img"].copy()
            pts_local = lidar_sample["pts"]
            lidar_ver = int(lidar_sample["version"])
            img_frame = int(cam_sample["frame"])
            lidar_frame = int(lidar_sample["frame"])
            img_ts = float(cam_sample["timestamp"])
            lidar_ts = float(lidar_sample["timestamp"])

        ts_delta = abs(float(img_ts) - float(lidar_ts))
        if pts_local is None or pts_local.shape[0] == 0:
            hist = np.zeros((LIDAR_H, LIDAR_W), dtype=np.float32)
        else:
            x, y = pts_local[:, 0], pts_local[:, 1]
            mask = (x > 0.0) & (x < HIST_X_MAX) & (y > HIST_Y_MIN) & (y < HIST_Y_MAX)

            if not np.any(mask):
                hist = np.zeros((LIDAR_H, LIDAR_W), dtype=np.float32)
            else:
                hist, _, _ = np.histogram2d(
                    x[mask], y[mask],
                    bins=(LIDAR_H, LIDAR_W),
                    range=[[HIST_X_MIN, HIST_X_MAX], [HIST_Y_MIN, HIST_Y_MAX]],
                )
                m = float(np.max(hist))
                if m > 0.0: hist = hist / (m + 1e-6)
                hist = hist.astype(np.float32, copy=False)

        with self._lock:
            if self._alive and lidar_ver != self._hist_version:
                np.copyto(self._latest_lidar_f32, hist)
                self._hist_version = lidar_ver
            self._record_alignment_accept(img_frame, lidar_frame, ts_delta)
            return (
                img,
                semantic,
                self._latest_lidar_f32.copy(),
                collision,
                img_frame,
                img_ts,
            )

    def clear_collision(self):
        with self._lock: self._latest_collision = 0.0

    def get_collision(self):
        with self._lock:
            return float(self._latest_collision)

    def destroy(self):
        with self._lock: self._alive = False
        for a in ("camera", "semantic_camera", "lidar", "collision_sensor"):
            try:
                actor = getattr(self, a, None)
                if actor is not None:
                    actor.stop()
                    actor.destroy()
            except Exception: pass

    def _attach_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(CAM_W))
        bp.set_attribute("image_size_y", str(CAM_H))
        bp.set_attribute("fov", str(CONFIG["sensors"]["camera"]["fov"]))
        bp.set_attribute("sensor_tick", str(CONFIG["carla"]["fixed_delta_seconds"]))

        tfm = carla.Transform(carla.Location(
            x=float(CONFIG["sensors"]["camera"]["mount"]["x"]),
            y=float(CONFIG["sensors"]["camera"]["mount"]["y"]),
            z=float(CONFIG["sensors"]["camera"]["mount"]["z"]),
        ))
        self.camera = self.world.spawn_actor(bp, tfm, attach_to=self.vehicle)
        self.camera.listen(self._camera_callback)

    def _attach_semantic_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(CAM_W))
        bp.set_attribute("image_size_y", str(CAM_H))
        bp.set_attribute("fov", str(CONFIG["sensors"]["camera"]["fov"]))
        bp.set_attribute("sensor_tick", str(CONFIG["carla"]["fixed_delta_seconds"]))

        tfm = carla.Transform(carla.Location(
            x=float(CONFIG["sensors"]["camera"]["mount"]["x"]),
            y=float(CONFIG["sensors"]["camera"]["mount"]["y"]),
            z=float(CONFIG["sensors"]["camera"]["mount"]["z"]),
        ))
        self.semantic_camera = self.world.spawn_actor(bp, tfm, attach_to=self.vehicle)
        self.semantic_camera.listen(self._semantic_callback)

    def _attach_lidar(self):
        bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        bp.set_attribute("range", str(CONFIG["sensors"]["lidar"]["range_m"]))
        bp.set_attribute("rotation_frequency", str(CONFIG["sensors"]["lidar"]["rotation_frequency_hz"]))
        bp.set_attribute("channels", str(CONFIG["sensors"]["lidar"]["channels"]))
        bp.set_attribute("points_per_second", str(LIDAR_PPS))
        bp.set_attribute("upper_fov", str(CONFIG["sensors"]["lidar"]["upper_fov_deg"]))
        bp.set_attribute("lower_fov", str(CONFIG["sensors"]["lidar"]["lower_fov_deg"]))
        bp.set_attribute("horizontal_fov", str(CONFIG["sensors"]["lidar"]["horizontal_fov_deg"]))
        bp.set_attribute("sensor_tick", str(CONFIG["carla"]["fixed_delta_seconds"]))

        tfm = carla.Transform(carla.Location(z=1.6))
        self.lidar = self.world.spawn_actor(bp, tfm, attach_to=self.vehicle)
        self.lidar.listen(self._lidar_callback)

    def _attach_collision_sensor(self):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._collision_callback)

    def _camera_callback(self, image: carla.Image):
        if not self._alive: return
        try:
            # CARLA sends BGRA
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            
            # Convert BGRA to RGB (strip Alpha and reverse channels 0-2)
            # This is the single most important mathematical fix for pre-trained weights
            rgb = raw[:, :, :3][:, :, ::-1] 
        except Exception: return

        with self._lock:
            if not self._alive: return
            np.copyto(self._latest_img_u8, rgb)
            self._has_img = True
            self._latest_img_frame = int(getattr(image, "frame", -1))
            self._latest_img_ts = float(getattr(image, "timestamp", 0.0))
            self._camera_buffer.append(
                {
                    "frame": self._latest_img_frame,
                    "timestamp": self._latest_img_ts,
                    "img": rgb.copy(),
                }
            )

    def _semantic_callback(self, image: carla.Image):
        if not self._alive: return
        try:
            image.convert(carla.ColorConverter.CityScapesPalette)
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            rgb = raw[:, :, :3][:, :, ::-1]
        except Exception: return

        with self._lock:
            if not self._alive: return
            np.copyto(self._latest_semantic_u8, rgb)
            self._has_semantic = True

    def _lidar_callback(self, data: carla.LidarMeasurement):
        if not self._alive: return
        try:
            pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
        except Exception: return

        n = min(int(pts.shape[0]), self._lidar_max_points)
        with self._lock:
            if not self._alive: return
            np.copyto(self._latest_lidar_pts[:n, :], pts[:n, :])
            self._latest_lidar_n = n
            self._has_lidar = True
            self._lidar_version += 1
            self._latest_lidar_frame = int(getattr(data, "frame", -1))
            self._latest_lidar_ts = float(getattr(data, "timestamp", 0.0))
            self._lidar_buffer.append(
                {
                    "frame": self._latest_lidar_frame,
                    "timestamp": self._latest_lidar_ts,
                    "pts": pts[:n, :].copy(),
                    "version": int(self._lidar_version),
                }
            )

    def _collision_callback(self, e: carla.CollisionEvent):
        if not self._alive: return
        imp = e.normal_impulse
        intensity = math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)
        with self._lock:
            if not self._alive: return
            self._latest_collision = float(min(intensity, 100.0))
