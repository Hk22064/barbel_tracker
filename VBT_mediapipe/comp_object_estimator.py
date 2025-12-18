import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from scipy.signal import savgol_filter
from enum import Enum, auto

# --- State Machine (Simplified from analyze_lift_scale.py) ---
class BarbellState(Enum):
    IDLE = auto()
    DESCENDING = auto()
    BOTTOM = auto()
    ASCENDING = auto()
    TOP = auto()

class PlateScaleAnalyzer:
    def __init__(self, plate_diameter_m=0.45):
        self.plate_diameter_m = plate_diameter_m
        self.scale_history = []
        self.global_scale = None
    
    def add_scale_sample(self, width_px):
        if width_px > 0:
            scale = self.plate_diameter_m / width_px
            self.scale_history.append(scale)
            
    def finalize_scale(self):
        if self.scale_history:
            self.global_scale = np.median(self.scale_history)
        else:
            self.global_scale = 0.0 # Fail
        return self.global_scale

class CompObjectEstimator:
    def __init__(self, model_path="Model_B_Clump_n/train/weights/best.pt", plate_diameter_cm=45.0):
        print(f"Loading Comparison Object Model: {model_path}...")
        try:
             self.model = YOLO(model_path)
             self.model_loaded = True
        except:
             print("Warning: Object Model not found. Comparison B will fail.")
             self.model = None
             self.model_loaded = False
             
        self.plate_diameter_m = plate_diameter_cm / 100.0
        self.tracker = None # We'll just use simple centroid tracking for this demo? 
                            # Or Kalman? Let's use simple frame-by-frame for simplicity in batch 
                            # (The original script had Kalman, but for batch stats, raw is okay-ish if filtered).
                            # Stick to simple midpoint tracking.
        
    def process_video_and_count(self, video_path):
        if not self.model_loaded:
            return 0, 0.0, [], []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine rotation for detection? (Same logic as main)
        # We will apply same rotation logic if needed, but let's assume the input video path is consistent.
        # Actually, we should handle rotation inside the reading loop.
        
        # For simplicity, we implement `analyze_lift_scale` logic:
        # 1. Collect trajectories & scales
        # 2. Smooth
        # 3. Count
        
        trajectory_y = []
        scale_analyzer = PlateScaleAnalyzer(self.plate_diameter_m)
        
        frame_count = 0
        start_time = fps # use current time?
        import time
        t0 = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Rotation check (Heuristic: Width > Height -> Rotate)
            # This must match the Hybrid script behavior for fair comparison!
            h, w = frame.shape[:2]
            if w > h:
                # Check filename keywords to avoid rotating actual landscape videos
                if "10rep" in video_path or "vertical" in video_path.lower():
                     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                     w, h = h, w # Swap dims
            
            results = self.model(frame, verbose=False, conf=0.25)
            
            # Find Plates (Class 0?)
            # Assuming Custom Model Class 0 = Plate/Weight
            plates = []
            for r in results:
                for box in r.boxes:
                    # if int(box.cls[0]) == 0: 
                    # Assuming all detected objects are relevant for this specific model
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    w_box = x2 - x1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Store
                    plates.append({'center': (center_x, center_y), 'width': w_box})
            
            # Simple Logic: Find pair of plates similar high/width? 
            # Or just take the average Y of all plates?
            # analyze_lift_scale used complex pair matching.
            # Simplified: Average Y of detected objects (robust enough for single lifter)
            
            avg_y = None
            if plates:
                y_sum = sum(p['center'][1] for p in plates)
                avg_y = y_sum / len(plates)
                
                # Scale Update
                # Use average width
                avg_w = sum(p['width'] for p in plates) / len(plates)
                scale_analyzer.add_scale_sample(avg_w)
            
            trajectory_y.append(avg_y)
            frame_count += 1
            
        cap.release()
        elapsed = time.time() - t0
        proc_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # --- Post Processing ---
        global_scale = scale_analyzer.finalize_scale()
        if global_scale == 0:
            return 0, proc_fps, [], [] # Fail
            
        # Interpolate None
        y_series = np.array(trajectory_y, dtype=np.float64)
        nans = np.isnan(y_series.astype(float)) # None becomes NaN in float array? No, object array.
        # Fix None handling
        y_floats = []
        for v in trajectory_y:
            y_floats.append(v if v is not None else np.nan)
        y_floats = np.array(y_floats)
        
        # Allow linear interp
        nans = np.isnan(y_floats)
        if np.all(nans):
             return 0, proc_fps, [], []

        x = np.flatnonzero(~nans)
        y_valid = y_floats[~nans]
        y_floats[nans] = np.interp(np.flatnonzero(nans), x, y_valid)
        
        # Smooth
        window = 15
        if len(y_floats) > window:
            y_smooth = savgol_filter(y_floats, window, 3)
        else:
            y_smooth = y_floats
            
        # Velocity
        velocities = []
        for i in range(1, len(y_smooth)):
            dy_px = y_smooth[i] - y_smooth[i-1]
            dist_m = abs(dy_px) * global_scale
            v = dist_m * fps
            velocities.append(v)
            
        # Filter Velocity
        if len(velocities) > 9:
            v_smooth = savgol_filter(velocities, 9, 2)
        else:
            v_smooth = velocities
            
        # Count Reps (Simple Crossing Logic or State Machine)
        # Use simple crossing for robustness in fallback
        # Count peaks > threshold?
        # Let's reuse the logic: Up-Down phases
        # Actually, let's just count significant peaks in Velocity curve?
        # Or position?
        # Using a simplified Peak Detection on Position (Top -> Bottom -> Top)
        
        rep_count = 0
        state = "TOP"
        start_y = y_smooth[0]
        thresh_m = 0.2 # 20cm range of motion min?
        thresh_px = thresh_m / global_scale
        
        bottom_y = start_y
        
        # Simple State Machine based on Position
        for y in y_smooth:
            if state == "TOP":
                if y > start_y + thresh_px: # Went down
                    state = "DESCENDING"
            elif state == "DESCENDING":
                if y > bottom_y:
                    bottom_y = y
                if y < bottom_y - (thresh_px * 0.5): # Started going up
                    state = "ASCENDING"
            elif state == "ASCENDING":
                if y < start_y + (thresh_px * 0.2): # Returned to top
                    rep_count += 1
                    state = "TOP"
                    bottom_y = y # reset
                    start_y = y # reset benchmark

        valid_times = [i/fps for i in range(len(v_smooth))]
        
        return rep_count, proc_fps, valid_times, list(v_smooth)
