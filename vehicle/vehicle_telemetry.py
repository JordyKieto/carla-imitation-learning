import math

import numpy as np
import carla

class VehicleTelemetry:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[INIT] Vehicle Telemetry - Zero-Centered Symmetric Mode")

    def telemetry_dict_to_vec(self, telem: dict, layout: list) -> np.ndarray:
        vec = []
        for k in layout:
            if k not in telem:
                # Fallback to 0.0 if a key is missing to prevent crashing the worker
                vec.append(np.array([0.0], dtype=np.float32))
            else:
                vec.append(np.array([telem[k]], dtype=np.float32))
        return np.concatenate(vec, axis=0)

    # ========================================================
    # Returns vehicle telemetry (Symmetric & Zero-Centered)
    # ========================================================
    def _compute_telemetry(self, telemetry_specs: dict, layout: list) -> np.ndarray:
            """
            Vehicle telemetry optimized for Flow Matching.
            Uses soft-normalization and zero-centering to prevent directional bias
            and ensure numerical stability in the Diffusion Transformer.
            """
            tr = self.vehicle.get_transform()
            vel = self.vehicle.get_velocity()
            acc = self.vehicle.get_acceleration()
            ctrl = self.vehicle.get_control()

            # 1. Physics in World Frame
            vx, vy, vz = float(vel.x), float(vel.y), float(vel.z)
            ax, ay, az = float(acc.x), float(acc.y), float(acc.z)
            
            # 2. Physics in Local Frame (Essential for Flow Matching)
            # Project velocity into car-relative Forward/Lateral space
            # CARLA uses a left-handed coordinate system.
            yaw_rad = math.radians(tr.rotation.yaw)
            sin_y = math.sin(yaw_rad)
            cos_y = math.cos(yaw_rad)
            
            # v_forward: Positive is forward, Negative is reverse
            v_forward = vx * cos_y + vy * sin_y
            # v_lateral: Positive is Right, Negative is Left
            v_lateral = -vx * sin_y + vy * cos_y

            out = {}

            # ----------------------------------------------------
            # STABILIZED PHYSICS (Zero-Centered / Soft-Saturated)
            # ----------------------------------------------------
            # Reference values for normalization
            REF_SPEED = 20.0  # Normalized speed where 20m/s maps to ~0.76 tanh
            REF_ACCEL = 5.0   # Normalized acceleration where 0.5G maps to ~0.76 tanh

            # 3. Scalar Magnitudes (Legacy Support)
            if telemetry_specs.get("speed_mps", False) or telemetry_specs.get("speed_kph", False):
                raw_speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                if telemetry_specs.get("speed_mps", False):
                    out["speed_mps"] = float(math.tanh(raw_speed / REF_SPEED))
                if telemetry_specs.get("speed_kph", False):
                    out["speed_kph"] = float(math.tanh((raw_speed * 3.6) / 72.0))

            if telemetry_specs.get("accel_mps2", False):
                raw_accel = math.sqrt(ax*ax + ay*ay + az*az)
                out["accel_mps2"] = float(math.tanh(raw_accel / REF_ACCEL))

            # ----------------------------------------------------
            # DIRECTIONAL COMPONENTS (The "Fix" for Steering Pinning)
            # ----------------------------------------------------
            # Explicitly provide directional velocity to resolve turn ambiguity
            out["v_forward"] = float(math.tanh(v_forward / REF_SPEED))
            out["v_lateral"] = float(math.tanh(v_lateral / REF_SPEED))

            # ----------------------------------------------------
            # PROPRIOCEPTION (Already Symmetric where possible)
            # ----------------------------------------------------
            if telemetry_specs.get("steering_angle", False):
                out["steering_angle"] = float(ctrl.steer) # Range [-1.0, 1.0]

            if telemetry_specs.get("throttle_input", False):
                # We use tanh(x*2) here to make the 0..1 range more expressive near zero
                out["throttle_input"] = float(math.tanh(ctrl.throttle * 2.0))

            if telemetry_specs.get("brake_input", False):
                out["brake_input"] = float(math.tanh(ctrl.brake * 2.0))

            # ----------------------------------------------------
            # EXTRAS (Normalized Angles)
            # ----------------------------------------------------
            if telemetry_specs.get("yaw_deg", False):
                # Map [-180, 180] to [-1.0, 1.0]
                out["yaw_deg"] = float(tr.rotation.yaw) / 180.0 

            if telemetry_specs.get("ego_xyz", False):
                # Note: These are unstable for long-distance training
                out["ego_x"] = float(tr.location.x) / 1000.0 
                out["ego_y"] = float(tr.location.y) / 1000.0
                out["ego_z"] = float(tr.location.z) / 1000.0

            return self.telemetry_dict_to_vec(out, layout)