import numpy as np
import time
from collections import deque

class VBTAnalyzer:
    """
    Analyzes movement data for Velocity Based Training (VBT).
    Handles calibration, velocity calculation, and rep analysis.
    """
    def __init__(self, smoothing_window=5):
        # Configuration
        self.smoothing_window = smoothing_window
        
        # State
        self.calibration_factor = None  # Meters per pixel
        self.prev_time = None
        self.prev_y = None
        
        # Data buffers
        self.velocity_buffer = deque(maxlen=smoothing_window)
        self.calibration_buffer = deque(maxlen=30) # Store recent wrist distances for auto-cal
        
        # Rep analysis state
        self.max_velocity_first_rep = None
        self.session_best_velocity = 0.0
        self.current_max_velocity = 0.0
        self.rep_velocities = [] # Store max velocity of each rep
        
        # State Machine
        self.STATE_WAITING = "WAITING"
        self.STATE_ECCENTRIC = "ECCENTRIC" # Going Down
        self.STATE_CONCENTRIC = "CONCENTRIC" # Going Up
        self.current_state = self.STATE_WAITING
        self.rep_count = 0
        
        # Thresholds
        self.velocity_threshold = 0.05 # m/s (Reverted to 0.05 to capture slow reps)
        self.min_concentric_displacement = 0.15 # m (15cm)
        self.min_peak_velocity = 0.15 # m/s
        
        self.concentric_start_y = None

    def attempt_auto_calibration(self, point1, point2, current_velocity_abs, real_distance_m=0.81):
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
                calc_factor = real_distance_m / mean_dist
                
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

    def calibrate(self, point1, point2, real_distance_m=0.81):
        """
        Manual Calibration (Legacy / Override).
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        dist_px = np.linalg.norm(p1 - p2)
        
        if dist_px > 0:
            calc_factor = real_distance_m / dist_px
            
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
        self.velocity_buffer.append(velocity_m_s)
        smoothed_velocity = sum(self.velocity_buffer) / len(self.velocity_buffer)
        
        return smoothed_velocity

    def process_rep(self, velocity):
        """
        State Machine for Rep Detection.
        WAITING -> ECCENTRIC (Neg Vel) -> CONCENTRIC (Pos Vel) -> WAITING
        """
        # Update Peak if in concentric phase
        if self.current_state == self.STATE_CONCENTRIC:
             if velocity > self.current_max_velocity:
                self.current_max_velocity = velocity

        # State Transitions
        if self.current_state == self.STATE_WAITING:
            # Start moving down (Eccentric)
            if velocity < -self.velocity_threshold:
                self.current_state = self.STATE_ECCENTRIC
                # print("State: ECCENTRIC")

        elif self.current_state == self.STATE_ECCENTRIC:
            # Switch to moving up (Concentric)
            if velocity > self.velocity_threshold:
                self.current_state = self.STATE_CONCENTRIC
                self.current_max_velocity = velocity # Reset peak for this rep
                self.concentric_start_y = self.prev_y # Record start point (Bottom)
                # print("State: CONCENTRIC")

        elif self.current_state == self.STATE_CONCENTRIC:
            # Finished moving up (Velocity drops to near zero or becomes negative)
            if velocity < self.velocity_threshold:
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
            
        return lines

