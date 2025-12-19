import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

class Yolo11MediaPipeEstimator:
    def __init__(self, model_path="yolo11n.pt", min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the Yolo11HybridEstimator.
        """
        print(f"Loading YOLO model: {model_path}...")
        self.yolo_model = YOLO(model_path)
        
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
        self.candidates = [] # Store candidate boxes during scanning phase

    def process_frame(self, frame):
        """
        Process a single frame.
        """
        landmarks_dict = {}
        self.frame_count += 1
        
        # 1. Determine Crop Box
        if self.locked_box is None:
            # Scanning Phase (First few frames)
            # Run YOLO
            yolo_results = self.yolo_model(frame, classes=[0], verbose=False) # class 0 = person
            
            target_box = None
            max_area = 0
            
            # Select the largest person
            if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    area = w * h
                    if area > max_area:
                        max_area = area
                        target_box = xyxy # [x1, y1, x2, y2]
            
            if target_box is not None:
                self.candidates.append(target_box)
            
            # Lock immediately on first valid detection
            # This is the "Pre-Stability Check" logic requested.
            
            if target_box is not None:
                h_img, w_img, _ = frame.shape
                x1, y1, x2, y2 = target_box
                box_w = x2 - x1
                box_h = y2 - y1
                
                # 40% margin to capture full motion (arms extending)
                # Especially important for side views (Right/Left) where arms move out of torso box.
                margin_x = int(box_w * 0.4)
                margin_y = int(box_h * 0.4)
                
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(w_img, x2 + margin_x)
                crop_y2 = min(h_img, y2 + margin_y)
                
                self.locked_box = [crop_x1, crop_y1, crop_x2, crop_y2]
                print(f"Target Locked! Crop: {self.locked_box}")
            else:
                 return landmarks_dict, None

        # 2. Use Locked Box
        if self.locked_box is None:
             return landmarks_dict, None
             
        crop_x1, crop_y1, crop_x2, crop_y2 = self.locked_box
        
        # Crop the image
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w, _ = cropped_frame.shape

        if crop_w == 0 or crop_h == 0:
            return landmarks_dict, None

        # 3. MediaPipe Pose Estimation
        # Convert BGR to RGB
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(cropped_rgb)

        # 4. Coordinate Transformation (Local -> Global)
        if pose_results.pose_landmarks:
            h_img, w_img, _ = frame.shape
            
            # Transform ALL landmarks (for drawing)
            for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                # Local coordinates (relative to crop)
                lx, ly = lm.x, lm.y
                
                # Transform to Global Pixel Coordinates
                gx_px = crop_x1 + (lx * crop_w)
                gy_px = crop_y1 + (ly * crop_h)
                
                # Update the landmark object itself to Global Normalized Coordinates
                lm.x = gx_px / w_img
                lm.y = gy_px / h_img
                
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
                # lm.x/y are now Global Normalized
                landmarks_dict[name] = [int(lm.x * w_img), int(lm.y * h_img)]
        
        # Add BBox to result for debugging
        landmarks_dict['bbox'] = self.locked_box

        return landmarks_dict, pose_results

    def reset(self):
        """
        Resets the locked box to trigger re-detection.
        """
        print("Resetting Target Lock...")
        self.locked_box = None

    def draw_landmarks(self, frame, results):
        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

    def draw_bbox(self, frame, bbox):
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red for Locked
            cv2.putText(frame, "LOCKED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def close(self):
        if self.pose:
            self.pose.close()
