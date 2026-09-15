import numpy as np
import math

class VehicleReward:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[INIT] Vehicle Reward System - Curvature & Anti-Spin Mode")
        
        # State tracking for chaining and smoothness
        self.chain_count = 0
        self.prev_steer = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0

    def compute_reward(self, collision_intensity, telemetry_vec, waypoints_raw):
        """
        Calculates a high-fidelity normalized reward.
        
        Args:
            telemetry_vec: [mps, kph, acc, v_fwd, v_lat, steer, thr, brk]
            waypoints_raw: (N, 3) local coords
        """
        # 1. CRITICAL FAILURES
        if collision_intensity > 0.0:
            self.chain_count = 0 
            return -1.0

        # --- Extract Telemetry ---
        speed_mps = float(telemetry_vec[0]) * 20.0 
        v_fwd = float(telemetry_vec[3])
        v_lat = float(telemetry_vec[4])
        steer = float(telemetry_vec[5]) # Current steering input (-1.0 to 1.0)
        
        # --- 1. CENTERING (Gaussian Falloff) ---
        pos_error = np.linalg.norm(waypoints_raw[0][:2])
        r_center = math.exp(-0.8 * (pos_error ** 2))

        # --- 2. WAYPOINT ALIGNMENT ---
        lookahead_idx = min(5, len(waypoints_raw) - 1)
        target_vec = waypoints_raw[lookahead_idx][:2]
        target_dist = np.linalg.norm(target_vec)
        
        r_align = 0.0
        if target_dist > 0.1:
            r_align = max(0.0, target_vec[0] / target_dist)

        # --- 3. DYNAMIC CURVATURE MATCHING (New) ---
        # Goal: Reward steering only when waypoints deviate from the forward axis (x).
        # We look at the average y-offset of the next 10 meters.
        r_curvature = 0.0
        if len(waypoints_raw) > 5:
            # Get the path curvature: how much do waypoints move left/right (y) as we go forward (x)?
            # Normalize by distance so sharp turns have higher 'required' steering
            path_y_offsets = waypoints_raw[:10, 1] 
            avg_path_y = np.mean(path_y_offsets)
            
            # Map path offset to a 'suggested' steering magnitude (0.0 to 1.0)
            desired_steer_sign = np.sign(avg_path_y)
            # A 2.0m lateral offset over 10m is a significant curve
            required_steer_intensity = np.clip(abs(avg_path_y) / 2.0, 0.0, 1.0)
            
            if required_steer_intensity > 0.05: # Only apply if there is an actual curve
                # Check if agent is steering in the correct direction
                # steer * desired_steer_sign will be positive if they match
                direction_match = steer * desired_steer_sign
                
                # Gaussian reward for how well the agent matches the intensity of the curve
                # If they steer too much or too little, the reward falls off.
                steer_error = abs(steer - (desired_steer_sign * required_steer_intensity))
                r_curvature = math.exp(-2.0 * steer_error) * required_steer_intensity
            else:
                # On straight roads, penalize excessive steering (anti-swerve)
                r_curvature = -0.1 * abs(steer)

        # --- 4. CHAINING & STABILITY ---
        if pos_error < 2.0 and r_align > 0.7:
            self.chain_count += 1
        else:
            self.chain_count = max(0, self.chain_count - 2)

        r_chain = min(0.3, 0.2 * math.log1p(self.chain_count / 10.0))
        r_progress = math.tanh(speed_mps / 7.0)
        
        side_slip = abs(v_lat) / (abs(v_fwd) + 1e-3)
        r_stability = math.exp(-2.0 * side_slip)

        # --- 5. SMOOTHNESS ---
        steer_delta = abs(steer - self.prev_steer)
        self.prev_steer = steer
        r_smooth = math.exp(-1.5 * steer_delta)

        # --- COMPOSITE SCORE ---
        # Weights: Added Curvature (15%), reduced Center/Align slightly to compensate
        # Weights: 30% Center, 30% Align, 15% Curvature, 15% Progress, 10% Smoothness
        base_score = (0.30 * r_center) + (0.30 * r_align) + \
                     (0.15 * r_curvature) + (0.15 * r_progress) + (0.10 * r_smooth)

        final_score = base_score * r_stability
        final_score += r_chain

        if speed_mps < 0.2: final_score = 0.0 
        
        return float(np.clip(final_score, -1.0, 1.5))