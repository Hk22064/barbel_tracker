import numpy as np
import time
from collections import deque
try:
    from scipy.signal import butter, lfilter, lfilter_zi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- Filter Strategies ---
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.buffer = deque(maxlen=window_size)
    
    def update(self, measurement):
        self.buffer.append(measurement)
        return sum(self.buffer) / len(self.buffer)

class KalmanFilter1D:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        # State [position, velocity] ... No, input is already velocity.
        # Simple 1D Kalman for a constant velocity model (assuming velocity doesn't change wildly instantly)
        self.x = 0.0 # Estimate
        self.p = 1.0 # Error Covariance
        self.q = process_noise # Process Noise
        self.r = measurement_noise # Measurement Noise
        # K = Gain

    def update(self, measurement):
        # Prediction (No control input, assume previous state)
        # x_pred = x
        p_pred = self.p + self.q
        
        # Update
        k = p_pred / (p_pred + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred
        return self.x

class ButterworthFilter:
    def __init__(self, order=2, cutoff=0.1, fps=30):
        if not SCIPY_AVAILABLE:
            print("[WARN] Scipy not found. Fallback to Moving Average.")
            self.filter = MovingAverageFilter(window_size=5)
            self.fallback = True
            return
        
        self.fallback = False
        nyquist = 0.5 * fps
        normal_cutoff = cutoff / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
        self.zi = lfilter_zi(self.b, self.a) * 0.0 # Initial state
        
    def update(self, measurement):
        if self.fallback: return self.filter.update(measurement)
        
        filtered, self.zi = lfilter(self.b, self.a, [measurement], zi=self.zi)
        return filtered[0]

class VBTAnalyzer:
    """
    Analyzes movement data for Velocity Based Training (VBT).
    Handles calibration, velocity calculation, and rep analysis.
    """
    def __init__(self, smoothing_window=5, velocity_threshold=0.05, filter_type="average", grip_finger="middle"):
        # Configuration
        self.smoothing_window = smoothing_window
        self.velocity_threshold = velocity_threshold # m/s
        self.filter_type = filter_type
        
        # Grip Calibration Logic
        # Assessment: Middle=0, Index=+4cm (Outer), Ring=-4cm (Inner), Pinky=-8cm (Inner)
        # Assuming ~2.0cm per finger width per hand.
        base_width = 0.81
        grip_offsets = {
            'index': 0.08,
            'middle': 0.04,
            'ring': 0.00,
            'pinky': -0.04
        }
        self.grip_width_m = base_width + grip_offsets.get(grip_finger.lower(), 0.0)
        
        print(f"Analyzer Config: Smooth={self.smoothing_window}, Threshold={self.velocity_threshold} m/s, Filter={self.filter_type}")
        print(f"Grip Calibration: Finger={grip_finger}, Effective Width={self.grip_width_m:.2f}m")
        
        # Initialize Filter Strategy
        if self.filter_type == "kalman":
            self.filter_strategy = KalmanFilter1D(process_noise=1e-3, measurement_noise=0.1) # Tuned strictly
        elif self.filter_type == "butterworth":
            # Assume ~30FPS, cutoff 3Hz (rapid movement allowed, noise cut)
            # Cutoff 0.1 (normalized) -> 0.1 * 15Hz = 1.5Hz? Too low.
            # Let's say freq=3Hz. 3/15 = 0.2.
            self.filter_strategy = ButterworthFilter(order=2, cutoff=3.0, fps=30) 
        else:
            self.filter_strategy = MovingAverageFilter(window_size=self.smoothing_window)
        
        # State
        self.calibration_factor = None  # Meters per pixel
        self.prev_time = None
        self.prev_y = None
        
        # Data buffers
        # self.velocity_buffer = deque(maxlen=smoothing_window) # NOW HANDLED BY STRATEGY
        self.calibration_buffer = deque(maxlen=30) # Store recent wrist distances for auto-cal
        
        # Rep analysis state
        self.max_velocity_first_rep = None
        self.session_best_velocity = 0.0
        self.session_best_velocity = 0.0
        self.current_max_velocity = 0.0
        self.rep_velocities = [] # Store max velocity of each rep
        self.rep_mean_velocities = [] # Store mean velocity of each rep
        self.current_rep_velocities = [] # Temp buffer for current rep
        
        # State Machine
        self.STATE_WAITING = "WAITING"
        self.STATE_ECCENTRIC = "ECCENTRIC" # Going Down
        self.STATE_CONCENTRIC = "CONCENTRIC" # Going Up
        self.current_state = self.STATE_WAITING
        self.rep_count = 0
        
        # Thresholds
        self.velocity_threshold = 0.05 # FIXED Entry Threshold (0.05 m/s)
        self.concentric_end_threshold = velocity_threshold # EXIT Threshold controlled by Slider
        self.min_concentric_displacement = 0.15 # m (15cm)
        self.min_peak_velocity = 0.15 # m/s
        
        print(f"Analyzer Config: Smooth={self.smoothing_window}, Entry={self.velocity_threshold}m/s, Exit={self.concentric_end_threshold}m/s, Filter={self.filter_type}")
        
        self.concentric_start_y = None

    def attempt_auto_calibration(self, point1, point2, current_velocity_abs):
        """
        Attempt to auto-calibrate if the user is standing still (low velocity) 
        and wrist distance is stable.
        """
        if self.calibration_factor is not None:
            return True # Already calibrated

        # If moving, reset buffer
        if current_velocity_abs > 0.1: # Threshold for "not standing still"
            self.calibration_buffer.clear()
            return False

        p1 = np.array(point1)
        p2 = np.array(point2)
        dist_px = np.linalg.norm(p1 - p2)
        
        self.calibration_buffer.append(dist_px)
        
        # Check stability if buffer is full (approx 1 second of data)
        if len(self.calibration_buffer) == self.calibration_buffer.maxlen:
            data = np.array(self.calibration_buffer)
            std_dev = np.std(data)
            mean_dist = np.mean(data)
            
            # If standard deviation is low (stable) and distance is reasonable
            if std_dev < 5.0 and mean_dist > 50: # 5px variation allowed
                calc_factor = self.grip_width_m / mean_dist
                
                # Check for Scale Explosion (too small pixel distance)
                # Relaxed to 0.1 to allow far subjects
                if calc_factor > 0.1:
                    print(f"[Auto-Calibration] WARNING: Calculated scale {calc_factor:.4f} is too high (Wrist Dist: {mean_dist:.1f}px). Using default 0.002.")
                    self.calibration_factor = 0.002
                else:
                    self.calibration_factor = calc_factor
                    print(f"[Auto-Calibration] Factor: {self.calibration_factor:.6f} m/px (Dist: {mean_dist:.1f}px)")
                return True
                
        return False

    def calibrate(self, point1, point2):
        """
        Manual Calibration (Legacy / Override).
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        dist_px = np.linalg.norm(p1 - p2)
        
        if dist_px > 0:
            calc_factor = self.grip_width_m / dist_px
            
            # Relaxed clamp to allow far subjects (e.g. 20px detected)
            # 20px -> 0.81/20 = 0.04. So limit needs to be > 0.04.
            if calc_factor > 0.1: # Very loose limit (Allow up to ~8px wrist dist)
                print(f"[Calibration] WARNING: Calculated scale {calc_factor:.4f} is too high (Dist: {dist_px:.1f}px). Using default 0.002.")
                self.calibration_factor = 0.002
            else:
                self.calibration_factor = calc_factor
                print(f"[Calibration] Factor: {self.calibration_factor:.6f} m/px (Dist: {dist_px:.1f}px)")
            return True
        return False

    def attempt_robust_calibration(self, landmarks, pose_results, mp_pose):
        """
        Robust calibration logic that handles side views and partial visibility.
        """
        if self.calibration_factor is not None:
            return

        if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            
            lw_x, lw_y = lw
            rw_x, rw_y = rw
            wrist_dist = np.sqrt((lw_x - rw_x)**2 + (lw_y - rw_y)**2)
            
            # Threshold: If wrist distance is too small (< 100px), assume Side View
            if wrist_dist > 100:
                self.calibrate(lw, rw) # Standard Wrist Calibration
            else:
                 # Fallback: Torso Calibration
                 use_default = True
                 
                 if pose_results and pose_results.pose_landmarks:
                    l_hip_vis = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].visibility
                    r_hip_vis = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility
                    
                    if l_hip_vis > 0.5 or r_hip_vis > 0.5:
                        if 'left_shoulder' in landmarks and 'left_hip' in landmarks:
                            ls = landmarks['left_shoulder']
                            lh = landmarks['left_hip']
                            torso_dist = np.sqrt((ls[0] - lh[0])**2 + (ls[1] - lh[1])**2)
                            
                            # Sanity Check for Hallucinations (e.g. right_9rep)
                            if torso_dist > 50:
                                self.calibration_factor = 0.50 / torso_dist
                                print(f"[Calibration] Side View (Torso Valid). Dist: {torso_dist:.1f}px. Factor: {self.calibration_factor:.4f}")
                                use_default = False
                            else:
                                print(f"[Calibration] Side View (Torso Invalid - Too Small). Dist: {torso_dist:.1f}px. Using Default.")

                 if use_default:
                      self.calibration_factor = 0.0015 # Fixed Conservative
                      print(f"[Calibration] Side View (Default). Factor: {self.calibration_factor:.4f}")

    def calculate_velocity(self, current_y, current_time):
        """
        Calculate vertical velocity (m/s).
        Positive Velocity = Upward (Lifting).
        Negative Velocity = Downward (Eccentric).
        """
        if self.calibration_factor is None:
            return 0.0
        
        if self.prev_time is None or self.prev_y is None:
            self.prev_time = current_time
            self.prev_y = current_y
            return 0.0
        
        dt = current_time - self.prev_time
        if dt <= 0:
            return 0.0
            
        # Vertical displacement (pixels). 
        # PrevY - CurrY means positive when moving UP (since Y is 0 at top)
        dy_px = self.prev_y - current_y 
        
        velocity_m_s = (dy_px * self.calibration_factor) / dt
        
        # Update previous state
        self.prev_time = current_time
        self.prev_y = current_y
        
        # Smoothing
        smoothed_velocity = self.filter_strategy.update(velocity_m_s)
        
        return smoothed_velocity

    def process_rep(self, velocity):
        """
        State Machine for Rep Detection.
        WAITING -> ECCENTRIC (Neg Vel) -> CONCENTRIC (Pos Vel) -> WAITING
        """
        # Update Peak and Buffer if in concentric phase
        if self.current_state == self.STATE_CONCENTRIC:
             if velocity > self.current_max_velocity:
                self.current_max_velocity = velocity
             
             # Log velocity for mean calculation
             if velocity > 0: # Only count positive VEL
                self.current_rep_velocities.append(velocity)

        # State Transitions
        if self.current_state == self.STATE_WAITING:
            # Start moving down (Eccentric)
            if velocity < -self.velocity_threshold:
                self.current_state = self.STATE_ECCENTRIC
                # print("State: ECCENTRIC")

        elif self.current_state == self.STATE_ECCENTRIC:
            # Switch to moving up (Concentric)
            if velocity > self.velocity_threshold: # Use strict Entry Threshold
                self.current_state = self.STATE_CONCENTRIC
                self.current_max_velocity = velocity # Reset peak for this rep
                self.current_rep_velocities = [velocity] # Reset & Start logging
                self.concentric_start_y = self.prev_y # Record start point (Bottom)
                # print("State: CONCENTRIC")

        elif self.current_state == self.STATE_CONCENTRIC:
            # Finished moving up (Velocity drops below LENIENT threshold)
            if velocity < self.concentric_end_threshold: # Use lenient Exit Threshold (e.g. 0.01)
                # Validate Rep (Noise Filter)
                is_valid = True
                
                # Validate Rep (Noise Filter)
                is_valid = True
                
                # 1. Displacement Check (MILD: 5cm)
                # Re-introduced to filter 2-3px noise (approx 0.5cm) while allowing short reps.
                if self.concentric_start_y is not None and self.prev_y is not None:
                     displacement_px = self.concentric_start_y - self.prev_y
                     displacement_m = displacement_px * self.calibration_factor
                     
                     if displacement_m < 0.05: # 5cm limit
                         is_valid = False
                         # print(f"Rep Ignored: Displacement {displacement_m:.3f}m < 0.05m")

                # 2. Peak Velocity Check
                if self.current_max_velocity < 0.1:
                     is_valid = False
                
                if is_valid:
                    self.commit_rep()
                
                self.current_state = self.STATE_WAITING
                self.concentric_start_y = None
                # print("State: WAITING (Rep Finished)")
        
        return self.current_max_velocity

    def commit_rep(self):
        """
        Finalize the rep data.
        """
        if self.current_max_velocity > 0:
            self.rep_count += 1
            self.rep_velocities.append(self.current_max_velocity)
            
            # Calculate Mean Velocity
            mean_vel = 0.0
            if self.current_rep_velocities:
                mean_vel = sum(self.current_rep_velocities) / len(self.current_rep_velocities)
            self.rep_mean_velocities.append(mean_vel)
            
            if self.max_velocity_first_rep is None:
                self.max_velocity_first_rep = self.current_max_velocity
            
            # Update Session Best
            if self.current_max_velocity > self.session_best_velocity:
                self.session_best_velocity = self.current_max_velocity
            
            # Reset
            # self.current_max_velocity = 0.0 # Keep it for display until next rep starts? 
            # Actually, let's keep it in 'rep_velocities' and reset current tracker when phase starts.
            
            return self.current_max_velocity
            
        return 0.0
    
    def get_fatigue_status(self):
        """
        Return fatigue percentage based on last committed rep vs first rep.
        """
        if self.max_velocity_first_rep is None or self.max_velocity_first_rep == 0:
            return 0.0
        
        if not self.rep_velocities:
            return 0.0
            
        last_rep_max = self.rep_velocities[-1]
        
        # Use session best as baseline
        baseline = self.session_best_velocity
        if baseline == 0:
            return 0.0
            
        drop_off = 1.0 - (last_rep_max / baseline)
        return max(0.0, drop_off * 100)

    def get_results_summary(self):
        """
        Generate a summary text report of the session.
        """
        lines = []
        lines.append("VBT Analysis Report")
        lines.append(f"Total Reps: {len(self.rep_velocities)}")
        lines.append("-" * 32)
        
        if not self.rep_velocities:
            lines.append("No reps detected.")
            return lines

        # Find session best (Absolute Peak of the set)
        session_best = max(self.rep_velocities) if self.rep_velocities else 0.0
        
        for i, vel in enumerate(self.rep_velocities):
            # Calc Drop off
            drop_off = 0.0
            if session_best > 0:
                drop_off = (1.0 - (vel / session_best)) * 100
            
            lines.append(f"Rep {i+1}: Peak: {vel:.2f} m/s, Drop-off: {drop_off:.1f}%")
            if i < len(self.rep_mean_velocities):
                 mean_v = self.rep_mean_velocities[i]
                 lines.append(f"        Mean: {mean_v:.2f} m/s")
            
        return lines

