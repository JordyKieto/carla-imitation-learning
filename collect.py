"""Automated data collection with the CARLA Traffic Manager autopilot as the expert.

For every spawn point on a map, finds which maneuvers (left / right / straight)
the next junction allows, then repeatedly drives the autopilot from a slightly
jittered start pose for --seconds, classifies what it actually did from the yaw
change, and keeps up to --captures-per-maneuver recordings per maneuver.

    python collect.py --map Town01
    python collect.py --map Town02 --seconds 20 --captures-per-maneuver 2 --seed 0

Writes recordings/drive_<timestamp>.tfrecord + .json summary + .route.json metadata.
"""
import argparse
import json
import logging
import os

import carla
import numpy as np
import pygame

from drive import DrivingInputContainer, TFRecordDriveRecorder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Collect")

MODEL_DRIVING_SPEED_DIFFERENCE_PERCENT = 60.0
MODEL_DRIVING_DISTANCE_TO_LEADING_VEHICLE = 2.0
GOLDEN_FIXED_SPEED_MPH = 18.0
GOLDEN_FIXED_SPEED_KMH = GOLDEN_FIXED_SPEED_MPH * 1.609344


class DisplayCamera:
    def __init__(self, world, vehicle, width, height):
        self.world = world
        self.vehicle = vehicle
        self.width = int(width)
        self.height = int(height)
        self.image = None
        self.actor = None
        self.surface = None
        self._attach()

    def _attach(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", "90")
        transform = carla.Transform(
            carla.Location(x=-8.0, z=3.5),
            carla.Rotation(pitch=-10.0),
        )
        self.actor = self.world.spawn_actor(
            bp,
            transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.SpringArmGhost,
        )
        self.actor.listen(self._on_image)

    def _on_image(self, image):
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        raw = raw.reshape((image.height, image.width, 4))
        self.image = raw[:, :, :3]

    def draw(self, display):
        if self.image is None:
            return
        frame = self.image[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        display.blit(self.surface, (0, 0))
        self.image = None

    def destroy(self):
        if self.actor is not None:
            try:
                self.actor.stop()
                self.actor.destroy()
            except Exception:
                pass
            self.actor = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--map", default="Town01")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=GOLDEN_FIXED_SPEED_KMH,
        help=(
            "Fixed native-autopilot desired speed in km/h when supported. "
            f"Default is {GOLDEN_FIXED_SPEED_MPH:.1f} mph."
        ),
    )
    parser.add_argument(
        "--traffic-manager-speed-difference",
        type=float,
        default=MODEL_DRIVING_SPEED_DIFFERENCE_PERCENT,
        help="Match model_driving.py Traffic Manager percentage speed difference.",
    )
    parser.add_argument("--maneuvers", default="left,right,straight")
    parser.add_argument("--record-dir", default="recordings")
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--ignore-traffic-lights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-stop-signs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-vehicles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-index", type=int, default=0, help="Start from spawn point index")
    parser.add_argument(
        "--max-attempts-per-spawn",
        type=int,
        default=20,
        help="Maximum native-autopilot respawns to try per spawn point.",
    )
    parser.add_argument(
        "--captures-per-maneuver",
        type=int,
        default=1,
        help="Number of accepted native-autopilot captures to keep per spawn/maneuver.",
    )
    parser.add_argument(
        "--yaw-jitter-deg",
        type=float,
        default=3.0,
        help="Max absolute yaw offset from the spawn, in degrees.",
    )
    parser.add_argument(
        "--lateral-jitter-m",
        type=float,
        default=0.25,
        help="Max absolute left/right offset from the spawn, in meters.",
    )
    parser.add_argument(
        "--forward-jitter-m",
        type=float,
        default=0.5,
        help="Max absolute forward/back offset from the spawn, in meters.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for spawn jitter")
    return parser.parse_args()


def reset_vehicle(vehicle, transform, world, tm_port):
    vehicle.set_autopilot(False, tm_port)
    vehicle.set_transform(transform)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    world.tick()


def jitter_start_transform(transform, rng, yaw_jitter_deg, lateral_jitter_m, forward_jitter_m):
    yaw_rad = np.deg2rad(transform.rotation.yaw)
    forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=np.float64)
    right = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)], dtype=np.float64)

    yaw_offset = float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)) if yaw_jitter_deg > 0.0 else 0.0
    lateral_offset = float(rng.uniform(-lateral_jitter_m, lateral_jitter_m)) if lateral_jitter_m > 0.0 else 0.0
    forward_offset = float(rng.uniform(-forward_jitter_m, forward_jitter_m)) if forward_jitter_m > 0.0 else 0.0
    offset_xy = (forward * forward_offset) + (right * lateral_offset)

    jittered = carla.Transform(
        carla.Location(
            x=transform.location.x + float(offset_xy[0]),
            y=transform.location.y + float(offset_xy[1]),
            z=transform.location.z,
        ),
        carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw + yaw_offset,
            roll=transform.rotation.roll,
        ),
    )
    jitter = {
        "yaw_offset_deg": yaw_offset,
        "lateral_offset_m": lateral_offset,
        "forward_offset_m": forward_offset,
    }
    return jittered, jitter


def write_route_metadata(record_path, metadata):
    meta_path = os.path.splitext(record_path)[0] + ".route.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _yaw_delta_deg(current_yaw, start_yaw):
    return (float(current_yaw) - float(start_yaw) + 180.0) % 360.0 - 180.0


def _classify_yaw_delta(delta, threshold_deg=25.0):
    if delta > threshold_deg:
        return "left"
    if delta < -threshold_deg:
        return "right"
    return "straight"


def classify_autopilot_trajectory(start_transform, yaw_history, left_right_threshold_deg=25.0):
    start_yaw = start_transform.rotation.yaw
    best_delta = 0.0
    for yaw in yaw_history:
        delta = _yaw_delta_deg(yaw, start_yaw)
        if abs(delta) > abs(best_delta):
            best_delta = delta
    return _classify_yaw_delta(best_delta, left_right_threshold_deg)


def _follow_waypoint_yaw(wp, step_m=2.0, steps=8):
    current = wp
    for _ in range(int(steps)):
        next_wps = current.next(float(step_m))
        if not next_wps:
            break
        current = next_wps[0]
    return current.transform.rotation.yaw


def spawn_available_maneuvers(carla_map, spawn_transform, requested_maneuvers):
    start_wp = carla_map.get_waypoint(
        spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if start_wp is None:
        return list(requested_maneuvers)

    current = start_wp
    branch_wps = []
    for _ in range(40):
        next_wps = current.next(2.0)
        if not next_wps:
            break
        if len(next_wps) > 1:
            branch_wps = next_wps
            break
        current = next_wps[0]

    if not branch_wps:
        branch_wps = current.next(2.0)

    options = set()
    reference_yaw = current.transform.rotation.yaw
    for branch_wp in branch_wps:
        branch_yaw = _follow_waypoint_yaw(branch_wp)
        maneuver = _classify_yaw_delta(_yaw_delta_deg(branch_yaw, reference_yaw))
        if maneuver in requested_maneuvers:
            options.add(maneuver)

    if not options and "straight" in requested_maneuvers:
        options.add("straight")
    return [m for m in requested_maneuvers if m in options]


def configure_native_autopilot(vehicle, tm, args):
    vehicle.set_autopilot(True, tm.get_port())
    if args.speed is not None:
        try:
            tm.set_desired_speed(vehicle, float(args.speed))
        except Exception:
            logger.warning(
                "[TM] set_desired_speed unavailable; Traffic Manager will use speed-limit-relative policy"
            )
    for method_name, value in (
        ("ignore_lights_percentage", 100.0 if args.ignore_traffic_lights else 0.0),
        ("ignore_signs_percentage", 100.0 if args.ignore_stop_signs else 0.0),
        ("ignore_vehicles_percentage", 100.0 if args.ignore_vehicles else 0.0),
    ):
        try:
            getattr(tm, method_name)(vehicle, float(value))
        except Exception:
            pass
    try:
        tm.auto_lane_change(vehicle, False)
    except Exception:
        pass


def write_buffered_recording(recorder, carla_map, spawn_idx, samples, metadata):
    recorder.start(carla_map.name, spawn_idx)
    for sample in samples:
        recorder.append(*sample)
    metadata["frames_written"] = len(samples)
    summary = recorder.stop(alignment_stats=metadata["alignment_stats"])
    if summary:
        write_route_metadata(summary["record_path"], metadata)
    return summary


def main():
    args = parse_args()
    maneuvers = [m.strip().lower() for m in args.maneuvers.split(",") if m.strip()]
    fixed_dt = 1.0 / float(args.fps)
    frames_per_route = max(1, int(round(float(args.seconds) * float(args.fps))))
    captures_per_maneuver = max(1, int(args.captures_per_maneuver))
    rng = np.random.default_rng(args.seed)

    pygame.init()
    display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Autopilot Data Collection")
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    logger.info("[CARLA] loading map=%s", args.map)
    world = client.load_world(args.map)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.traffic_manager_port)
    tm.set_synchronous_mode(True)
    tm.global_percentage_speed_difference(float(args.traffic_manager_speed_difference))
    tm.set_global_distance_to_leading_vehicle(MODEL_DRIVING_DISTANCE_TO_LEADING_VEHICLE)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available")

    bp = world.get_blueprint_library().find(args.vehicle_filter)
    vehicle = world.spawn_actor(bp, spawn_points[0])
    vehicle.set_autopilot(False, tm.get_port())

    display_camera = DisplayCamera(world, vehicle, args.width, args.height)
    collector = DrivingInputContainer(world=world, vehicle=vehicle)
    recorder = TFRecordDriveRecorder(args.record_dir)

    logger.info(
        "[START] spawns=%d maneuvers=%s seconds=%.1f frames=%d control=native_autopilot "
        "tm_desired_speed=%.1fkm/h tm_speed_difference=%.1f captures_per_maneuver=%d max_attempts_per_spawn=%d "
        "yaw_jitter=%.1fdeg lateral_jitter=%.2fm forward_jitter=%.2fm seed=%s",
        len(spawn_points),
        ",".join(maneuvers),
        args.seconds,
        frames_per_route,
        float(args.speed) if args.speed is not None else -1.0,
        float(args.traffic_manager_speed_difference),
        captures_per_maneuver,
        int(args.max_attempts_per_spawn),
        float(args.yaw_jitter_deg),
        float(args.lateral_jitter_m),
        float(args.forward_jitter_m),
        args.seed,
    )

    try:
        for spawn_idx, spawn in enumerate(spawn_points):
            if spawn_idx < args.start_index:
                continue
            spawn_maneuvers = spawn_available_maneuvers(carla_map, spawn, maneuvers)
            if not spawn_maneuvers:
                logger.info("[SKIP] spawn=%d no requested native-autopilot options found", spawn_idx)
                continue
            logger.info(
                "[SPAWN] spawn=%d target_maneuvers=%s",
                spawn_idx,
                ",".join(spawn_maneuvers),
            )
            captured_counts = {m: 0 for m in spawn_maneuvers}
            max_attempts = max(1, int(args.max_attempts_per_spawn))

            for attempt_idx in range(max_attempts):
                missing = [m for m in spawn_maneuvers if captured_counts[m] < captures_per_maneuver]
                if not missing:
                    break

                start_transform, jitter = jitter_start_transform(
                    spawn,
                    rng,
                    float(args.yaw_jitter_deg),
                    float(args.lateral_jitter_m),
                    float(args.forward_jitter_m),
                )
                reset_vehicle(vehicle, start_transform, world, tm.get_port())
                collector.clear_waypoint_route()
                collector.reset_alignment_stats()
                configure_native_autopilot(vehicle, tm, args)

                logger.info(
                    "[ATTEMPT] spawn=%d attempt=%d/%d missing=%s frames=%d "
                    "yaw=%+.1fdeg lateral=%+.2fm forward=%+.2fm",
                    spawn_idx,
                    attempt_idx + 1,
                    max_attempts,
                    ",".join(missing),
                    frames_per_route,
                    jitter["yaw_offset_deg"],
                    jitter["lateral_offset_m"],
                    jitter["forward_offset_m"],
                )

                samples = []
                yaw_history = []
                for frame_idx in range(frames_per_route):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt

                    world.tick()
                    yaw_history.append(vehicle.get_transform().rotation.yaw)

                    sample = collector.collect_step()
                    if sample is not None:
                        samples.append(sample)

                    if frame_idx % max(1, int(args.fps)) == 0:
                        logger.info(
                            "[STEP] spawn=%d/%d attempt=%d/%d frame=%d/%d buffered=%d",
                            spawn_idx,
                            len(spawn_points) - 1,
                            attempt_idx + 1,
                            max_attempts,
                            frame_idx,
                            frames_per_route,
                            len(samples),
                        )

                    display_camera.draw(display)
                    collector._debug_waypoints(life_time=0.15, size=0.12)
                    pygame.display.update()
                    clock.tick_busy_loop(int(args.fps))

                vehicle.set_autopilot(False, tm.get_port())
                if not samples:
                    logger.info(
                        "[DISCARD] spawn=%d attempt=%d no sensor samples",
                        spawn_idx,
                        attempt_idx + 1,
                    )
                    continue

                maneuver = classify_autopilot_trajectory(start_transform, yaw_history)
                logger.info(
                    "[CLASSIFY] spawn=%d attempt=%d maneuver=%s targets=%s counts=%s",
                    spawn_idx,
                    attempt_idx + 1,
                    maneuver,
                    ",".join(spawn_maneuvers),
                    ",".join(f"{m}:{captured_counts[m]}/{captures_per_maneuver}" for m in spawn_maneuvers),
                )
                if maneuver not in spawn_maneuvers:
                    logger.info(
                        "[DISCARD] spawn=%d attempt=%d maneuver=%s not a target for this spawn",
                        spawn_idx,
                        attempt_idx + 1,
                        maneuver,
                    )
                    continue
                if captured_counts[maneuver] >= captures_per_maneuver:
                    logger.info(
                        "[DISCARD] spawn=%d attempt=%d maneuver=%s over capture target",
                        spawn_idx,
                        attempt_idx + 1,
                        maneuver,
                    )
                    continue

                metadata = {
                    "map": carla_map.name,
                    "spawn_index": spawn_idx,
                    "attempt_index": attempt_idx,
                    "maneuver": maneuver,
                    "maneuver_capture_index": captured_counts[maneuver],
                    "captures_per_maneuver": captures_per_maneuver,
                    "spawn_target_maneuvers": spawn_maneuvers,
                    "seconds": args.seconds,
                    "fps": args.fps,
                    "frames_requested": frames_per_route,
                    "control_policy": "native_traffic_manager_autopilot",
                    "classification_policy": "max_abs_yaw_delta_threshold",
                    "traffic_manager_desired_speed_kmh": (
                        float(args.speed) if args.speed is not None else None
                    ),
                    "traffic_manager_speed_difference": float(args.traffic_manager_speed_difference),
                    "traffic_manager_distance_to_leading_vehicle": MODEL_DRIVING_DISTANCE_TO_LEADING_VEHICLE,
                    "start_jitter": jitter,
                    "start_transform": {
                        "x": start_transform.location.x,
                        "y": start_transform.location.y,
                        "z": start_transform.location.z,
                        "pitch": start_transform.rotation.pitch,
                        "yaw": start_transform.rotation.yaw,
                        "roll": start_transform.rotation.roll,
                    },
                    "alignment_stats": collector.get_alignment_stats(),
                }
                write_buffered_recording(recorder, carla_map, spawn_idx, samples, metadata)
                captured_counts[maneuver] += 1
                logger.info(
                    "[CAPTURED] spawn=%d maneuver=%s counts=%s",
                    spawn_idx,
                    maneuver,
                    ",".join(f"{m}:{captured_counts[m]}/{captures_per_maneuver}" for m in spawn_maneuvers),
                )

            missing = [m for m in spawn_maneuvers if captured_counts[m] < captures_per_maneuver]
            if missing:
                logger.info(
                    "[SKIP] spawn=%d missing maneuvers after %d attempts: %s",
                    spawn_idx,
                    max_attempts,
                    ",".join(missing),
                )

            vehicle.set_autopilot(False, tm.get_port())
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            world.tick()

    except KeyboardInterrupt:
        logger.info("[STOP] interrupted")
    finally:
        if recorder.recording:
            recorder.stop(alignment_stats=collector.get_alignment_stats())
        collector.destroy()
        display_camera.destroy()
        vehicle.destroy()
        world.apply_settings(original_settings)
        pygame.quit()


if __name__ == "__main__":
    main()
