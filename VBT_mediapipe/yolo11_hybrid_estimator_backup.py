import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

class Yolo11HybridEstimator:
    def __init__(self, model_path="yolo11n.pt", min_detection_confidence=0.5, min_tracking_confidence=0.5, device=None):
        """
        Initializes the Yolo11HybridEstimator.
        device: 'cpu' or 'cuda' (or None for auto)
        """
        print(f"Loading YOLO model: {model_path} (Device: {device})...")
        self.yolo_model = YOLO(model_path)
        self.device = device
        
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, # Video Mode for smoothing
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Lock State
        self.locked_box = None
        self.frame_count = 0
        self.scan_interval = 30 # Check every ~1 second (assuming 30fps)
        self.prev_scan_box = None
        self.current_crop_box = None # Active crop box (might be locked or dynamic)
        
        # Stability Config
        self.stability_threshold = 50 # Pixel difference threshold for locking

    def process_frame(self, frame):
        """
        Process a single frame.
        """
        landmarks_dict = {}
        self.frame_count += 1
        h_img, w_img, _ = frame.shape
        
        # 1. Determine Crop Box (Stable Lock Logic)
        
        # If already locked, use it
        if self.locked_box is not None:
             self.current_crop_box = self.locked_box
             
        # If not locked, check if we should scan (Every 30 frames or if no crop yet)
        elif self.current_crop_box is None or self.frame_count % self.scan_interval == 0:
            
            # Run YOLO
            yolo_results = self.yolo_model(frame, classes=[0], verbose=False, device=self.device)
            target_box = None
            max_area = 0
            
            # Select largest person
            if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    area = w * h
                    if area > max_area:
                        max_area = area
                        target_box = xyxy
            
            # Process Detection
            if target_box is not None:
                x1, y1, x2, y2 = target_box
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Add Margin (20%)
                margin_x = int(box_w * 0.2)
                margin_y = int(box_h * 0.2)
                
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(w_img, x2 + margin_x)
                crop_y2 = min(h_img, y2 + margin_y)
                new_crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
                
                # Check Stability
                if self.prev_scan_box is not None:
                    # Calculate difference (center distance + size diff)
                    p_x1, p_y1, p_x2, p_y2 = self.prev_scan_box
                    n_x1, n_y1, n_x2, n_y2 = new_crop_box
                    
                    diff = abs(p_x1 - n_x1) + abs(p_y1 - n_y1) + abs(p_x2 - n_x2) + abs(p_y2 - n_y2)
                    
                    if diff < self.stability_threshold:
                        # Stable! Lock it.
                        self.locked_box = new_crop_box
                        print(f"[{self.frame_count}] Stable Lock Engaged! Diff: {diff}")
                    else:
                        print(f"[{self.frame_count}] Unstable. Diff: {diff}. Updating dynamic box.")
                
                # Update State
                self.prev_scan_box = new_crop_box
                self.current_crop_box = new_crop_box
                
            else:
                 # Detection failed this frame, keep using old box if exists, else return
                 if self.current_crop_box is None:
                     return landmarks_dict, None

        # 2. Use Current Box
        if self.current_crop_box is None:
             return landmarks_dict, None
             
        crop_x1, crop_y1, crop_x2, crop_y2 = self.current_crop_box
        
        # Crop
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w, _ = cropped_frame.shape

        if crop_w == 0 or crop_h == 0:
            return landmarks_dict, None

        # 3. MediaPipe Pose Estimation
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(cropped_rgb)

        # 4. Coordinate Transformation
        if pose_results.pose_landmarks:
            landmark_names = {
                self.mp_pose.PoseLandmark.LEFT_WRIST: 'left_wrist',
                self.mp_pose.PoseLandmark.RIGHT_WRIST: 'right_wrist',
                self.mp_pose.PoseLandmark.LEFT_SHOULDER: 'left_shoulder',
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER: 'right_shoulder',
                self.mp_pose.PoseLandmark.LEFT_ELBOW: 'left_elbow',
                self.mp_pose.PoseLandmark.RIGHT_ELBOW: 'right_elbow',
                self.mp_pose.PoseLandmark.LEFT_HIP: 'left_hip',
                self.mp_pose.PoseLandmark.RIGHT_HIP: 'right_hip',
                self.mp_pose.PoseLandmark.NOSE: 'nose'
            }
            
            for lm_id, name in landmark_names.items():
                lm = pose_results.pose_landmarks.landmark[lm_id]
                landmarks_dict[name] = [int(lm.x * w_img), int(lm.y * h_img)]
        
        landmarks_dict['bbox'] = self.current_crop_box
        landmarks_dict['locked'] = (self.locked_box is not None)

        return landmarks_dict, pose_results

    def reset(self):
        print("Resetting Target Lock...")
        self.locked_box = None
        self.current_crop_box = None
        self.prev_scan_box = None
        self.frame_count = 0

    def draw_landmarks(self, frame, results):
        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

    def draw_bbox(self, frame, bbox, locked=False):
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) # Green for Dynamic
            label = "Tracking"
            if locked:
                color = (0, 0, 255) # Red for Locked
                label = "LOCKED"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def close(self):
        if self.pose:
            self.pose.close()
