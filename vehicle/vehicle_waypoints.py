import carla
import numpy as np
import math

from utils.config import CONFIG

WAYPOINT_PATH_LEN = CONFIG["waypoints"]["path_len"]

class Waypoints:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[INIT] Waypoints Generator (Full 18-Channel | Stabilized)")
        
        # Internal storage: (N, 3) -> [Local_X, Local_Y, Local_Z] 
        self._last_raw_local_xyz = None
        self._waypoint_route = None
        
        # State tracking for signal derivation
        self._prev_wp_headings = None
        self._prev_wp_curvature = None
        self._prev_wp_turn_severity = None
        self._prev_wp_lateral_offset = None
        self._prev_wp_dist_norm = None

    def set_waypoint_route(self, route):
        """
        Sets an explicit route for future-waypoint labels.
        Route items may be carla.Waypoint or (carla.Waypoint, RoadOption).
        """
        self._waypoint_route = list(route) if route else None
        self.reset_waypoint_state()

    def clear_waypoint_route(self):
        self._waypoint_route = None
        self.reset_waypoint_state()

    def reset_waypoint_state(self):
        self._last_raw_local_xyz = None
        self._prev_wp_headings = None
        self._prev_wp_curvature = None
        self._prev_wp_turn_severity = None
        self._prev_wp_lateral_offset = None
        self._prev_wp_dist_norm = None

    @staticmethod
    def _route_item_waypoint(item):
        if isinstance(item, (tuple, list)) and item:
            return item[0]
        return item

    def _route_local_coords(self, route, step_m):
        if not route:
            return []

        v_tf = self.vehicle.get_transform()
        v_loc = v_tf.location
        waypoints = [self._route_item_waypoint(item) for item in route]
        waypoints = [wp for wp in waypoints if wp is not None]
        if not waypoints:
            return []

        nearest_idx = min(
            range(len(waypoints)),
            key=lambda i: waypoints[i].transform.location.distance(v_loc),
        )

        local_coords = []
        last_loc = None
        for wp in waypoints[nearest_idx:]:
            loc = wp.transform.location
            if last_loc is not None and loc.distance(last_loc) < step_m:
                continue
            loc_relative = v_tf.inverse_transform(loc)
            local_coords.append([loc_relative.x, loc_relative.y, loc_relative.z])
            last_loc = loc
            if len(local_coords) >= WAYPOINT_PATH_LEN:
                break
        return local_coords

    def _format_waypoint_coords(self, local_coords):
        while len(local_coords) < WAYPOINT_PATH_LEN:
            local_coords.append(local_coords[-1] if local_coords else [0, 0, 0])

        coords = np.array(local_coords[:WAYPOINT_PATH_LEN], dtype=np.float32)
        self._last_raw_local_xyz = coords

        raw_out = np.zeros((WAYPOINT_PATH_LEN, 3), dtype=np.float32)
        dist = np.linalg.norm(coords[:, :2], axis=1)
        safe_dist = np.where(dist < 1e-4, 1.0, dist)

        raw_out[:, 0] = coords[:, 0] / safe_dist
        raw_out[:, 1] = coords[:, 1] / safe_dist
        raw_out[dist < 1e-4, 1] = 0.0
        raw_out[:, 2] = np.clip(dist, 0.0, 50.0)

        norm_out = raw_out.copy()
        norm_out[:, 2] /= 50.0
        return norm_out, raw_out

    def _future_waypoint_path(self, step_m=2.0):
        """
        Calculates path in local car space for stability.
        Returns: (normalized_wp, raw_wp)
        """
        if self._waypoint_route:
            local_coords = self._route_local_coords(self._waypoint_route, step_m)
            if local_coords:
                return self._format_waypoint_coords(local_coords)

        carla_map = self.world.get_map()
        v_tf = self.vehicle.get_transform()
        v_loc = v_tf.location

        # Strict lane projection to ensure we don't snap to sidewalks
        wp = carla_map.get_waypoint(v_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if wp is None:
            empty = np.zeros((WAYPOINT_PATH_LEN, 3), dtype=np.float32)
            self._last_raw_local_xyz = None
            return empty, empty

        local_coords = []
        for i in range(WAYPOINT_PATH_LEN):
            loc_relative = v_tf.inverse_transform(wp.transform.location)
            local_coords.append([loc_relative.x, loc_relative.y, loc_relative.z])
            
            # Simple linear lookahead
            next_wps = wp.next(step_m)
            if next_wps:
                wp = next_wps[0]
            else:
                break

        # Padding
        while len(local_coords) < WAYPOINT_PATH_LEN:
            local_coords.append(local_coords[-1] if local_coords else [0, 0, 0])

        return self._format_waypoint_coords(local_coords)

    def _debug_waypoints(self, life_time=3, size=0.15):
        if self._last_raw_local_xyz is None: return

        v_tf = self.vehicle.get_transform()
        dbg = self.world.debug
        green = carla.Color(0, 255, 0)

        for i in range(len(self._last_raw_local_xyz)):
            loc_rel = self._last_raw_local_xyz[i]
            world_loc = v_tf.transform(carla.Location(
                x=float(loc_rel[0]), 
                y=float(loc_rel[1]), 
                z=float(loc_rel[2]) + 1.2 
            ))
            dbg.draw_point(world_loc, size=size, color=green, life_time=life_time)
            # dbg.draw_hud_draw_point(world_loc, size=size, color=green, life_time=life_time)


    def _derive_waypoint_signal(self):
        """
        FULL 18-Channel Signal Extraction (Stabilized).
        Includes velocity gating to prevent noise when stopped.
        """
        if self._last_raw_local_xyz is None:
            return np.zeros((WAYPOINT_PATH_LEN, 18), dtype=np.float32)

        coords = self._last_raw_local_xyz
        dist = np.linalg.norm(coords[:, :2], axis=1)
        
        # Safe denominators
        safe_dist = np.where(dist < 1e-4, 1.0, dist)
        dir_x = coords[:, 0] / safe_dist
        dir_y = coords[:, 1] / safe_dist
        
        N = WAYPOINT_PATH_LEN
        progress = np.linspace(0.0, 1.0, N, dtype=np.float32)
        
        # 1. Calculate Headings
        headings = np.arctan2(dir_y, dir_x).astype(np.float32)

        # 2. Curvature & Heading (Stabilized)
        d_heading = np.diff(headings, prepend=headings[0])
        # Wrap angles to [-pi, pi] to handle 360->0 transitions
        d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi
        
        d_dist = np.diff(dist, prepend=dist[0])
        
        # FIX: VELOCITY GATING
        # If the distance between points is tiny (car stopped or points stacked),
        # the curvature calculation (d_angle / d_dist) explodes to Infinity.
        # We mask these out.
        valid_motion_mask = d_dist > 1e-3
        
        raw_curvature = np.zeros_like(d_heading)
        np.divide(d_heading, d_dist, out=raw_curvature, where=valid_motion_mask)
        
        curvature = np.clip(raw_curvature, -10.0, 10.0)
        
        curvature_change = np.diff(curvature, prepend=curvature[0])
        curvature_change = np.clip(curvature_change, -5.0, 5.0)
        
        heading_accum = np.cumsum(d_heading)

        # 3. Severity & Offsets
        lookahead = 5
        turn_severity = np.zeros(N, dtype=np.float32)
        for i in range(N):
            val = np.max(np.abs(curvature[i : i+lookahead]))
            turn_severity[i] = float(val)
        
        turn_severity = np.clip(turn_severity, 0.0, 10.0)

        # FIX: Lateral Deadzone
        # If we are < 10cm off center, report 0 offset to stop micro-steering
        raw_lat = (dir_y * dist)
        lat_deadzone = np.where(np.abs(raw_lat) < 0.1, 0.0, raw_lat)
        lateral_offset = np.clip(lat_deadzone / 5.0, -1.0, 1.0)
        
        dist_norm = np.clip(dist / 50.0, 0.0, 1.0)

        # 4. Temporal Deltas
        prev_valid = 1.0 if self._prev_wp_headings is not None else 0.0
        d_h_norm, d_curv, d_sev, d_lat, d_dist_n = [np.zeros(N) for _ in range(5)]

        if prev_valid:
            try:
                dh = headings - self._prev_wp_headings
                d_h_norm = ((dh + np.pi) % (2.0 * np.pi) - np.pi) / np.pi
                d_curv = curvature - self._prev_wp_curvature
                d_sev = turn_severity - self._prev_wp_turn_severity
                d_lat = lateral_offset - self._prev_wp_lateral_offset
                d_dist_n = dist_norm - self._prev_wp_dist_norm
                
                # Clip deltas
                d_curv = np.clip(d_curv, -5.0, 5.0)
                d_sev = np.clip(d_sev, -5.0, 5.0)
            except:
                prev_valid = 0.0

        self._prev_wp_headings = headings.copy()
        self._prev_wp_curvature = curvature.copy()
        self._prev_wp_turn_severity = turn_severity.copy()
        self._prev_wp_lateral_offset = lateral_offset.copy()
        self._prev_wp_dist_norm = dist_norm.copy()

        # 5. Ego State
        ctrl = self.vehicle.get_control()
        vel = self.vehicle.get_velocity()
        ang = self.vehicle.get_angular_velocity()
        v_tf = self.vehicle.get_transform()
        
        rad = math.radians(v_tf.rotation.yaw)
        raw_lat_vel = -math.sin(rad) * vel.x + math.cos(rad) * vel.y
        
        # Stabilized Heading Error
        if N > 1 and dist[1] > 0.1:
            heading_err = math.atan2(coords[1,1]-coords[0,1], coords[1,0]-coords[0,0])
        else:
            heading_err = 0.0

        c_steer = np.full(N, ctrl.steer)
        c_cte = np.full(N, np.clip(lat_deadzone[0] / 2.0, -1.0, 1.0)) # Use deadzone here too
        c_lat_v = np.full(N, np.clip(raw_lat_vel / 5.0, -1.0, 1.0))
        c_yaw = np.full(N, np.clip(math.radians(ang.z), -1.0, 1.0))
        c_head_err = np.full(N, np.clip(heading_err / (math.pi/2), -1.0, 1.0))

        # 6. Final Stack
        return np.stack([
            progress, curvature, curvature_change, heading_accum, turn_severity, # 0-4
            lateral_offset, dist_norm, # 5-6
            d_h_norm, d_curv, d_sev, d_lat, d_dist_n, # 7-11
            np.full(N, prev_valid), # 12
            c_steer, c_cte, c_lat_v, c_yaw, c_head_err # 13-17
        ], axis=1).astype(np.float32)
