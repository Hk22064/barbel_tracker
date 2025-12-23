import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

class YoloHybridEstimator:
    """
    Hybrid Pose Estimator:
    1. Detect Person using YOLOv8.
    2. Crop the person area.
    3. Estimate Pose using MediaPipe on the crop.
    4. Transform coordinates back to global frame.
    """
    def __init__(self, yolo_model='models/yolo11s.pt', static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # 1. Initialize YOLO
        print(f"Loading YOLO model: {yolo_model}...")
        self.yolo = YOLO(yolo_model)
        
        # 2. Initialize MediaPipe Pose
        # Note: static_image_mode=True is recommended for cropped independent frames
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Landmark Names Mappings (Same as PoseEstimator)
        self.LANDMARK_NAMES = {
            self.mp_pose.PoseLandmark.LEFT_WRIST: 'left_wrist',
            self.mp_pose.PoseLandmark.RIGHT_WRIST: 'right_wrist',
            self.mp_pose.PoseLandmark.LEFT_ELBOW: 'left_elbow',
            self.mp_pose.PoseLandmark.RIGHT_ELBOW: 'right_elbow',
            self.mp_pose.PoseLandmark.LEFT_SHOULDER: 'left_shoulder',
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER: 'right_shoulder',
            self.mp_pose.PoseLandmark.LEFT_HIP: 'left_hip',
            self.mp_pose.PoseLandmark.RIGHT_HIP: 'right_hip',
            self.mp_pose.PoseLandmark.LEFT_KNEE: 'left_knee',
            self.mp_pose.PoseLandmark.RIGHT_KNEE: 'right_knee',
            self.mp_pose.PoseLandmark.LEFT_ANKLE: 'left_ankle',
            self.mp_pose.PoseLandmark.RIGHT_ANKLE: 'right_ankle',
            self.mp_pose.PoseLandmark.NOSE: 'nose'
        }

    def process_frame(self, frame):
        """
        Process the frame using YOLO -> Crop -> MediaPipe pipeline.
        
        Returns:
            landmarks_dict (dict): Global pixel coordinates of keypoints.
            results (object): MediaPipe results object with GLOBAL NORMALIZED coordinates (for drawing).
        """
        h_frame, w_frame, _ = frame.shape
        
        # ---------------------------------------------------------
        # 1. Object Detection (YOLOv8)
        # ---------------------------------------------------------
        # Run inference, class=0 is 'person'
        yolo_results = self.yolo(frame, classes=[0], verbose=False)
        
        best_box = None
        
        # Find the best person (Highest Confidence or Max Area)
        # yolo_results is a list (one per image), we only provided one frame
        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes
            
            # Simple strategy: Choose the one with highest confidence
            # (Alternative: choose largest area if multiple people are present)
            best_idx = np.argmax(boxes.conf.cpu().numpy())
            best_box = boxes.xyxy[best_idx].cpu().numpy() # [x1, y1, x2, y2]
            
        if best_box is None:
            # No person detected
            return {}, None

        # ---------------------------------------------------------
        # 2. Crop & Pad
        # ---------------------------------------------------------
        x1, y1, x2, y2 = best_box
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Add Margin (10%)
        margin_x = box_w * 0.1
        margin_y = box_h * 0.1
        
        crop_x1 = int(max(0, x1 - margin_x))
        crop_y1 = int(max(0, y1 - margin_y))
        crop_x2 = int(min(w_frame, x2 + margin_x))
        crop_y2 = int(min(h_frame, y2 + margin_y))
        
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        if crop_w <= 0 or crop_h <= 0:
             return {}, None
             
        # Extract the crop
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Optim: Convert to RGB once here
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------
        # 3. Pose Estimation (MediaPipe)
        # ---------------------------------------------------------
        results = self.pose.process(cropped_rgb)
        
        landmarks_dict = {}
        
        if results.pose_landmarks:
            # ---------------------------------------------------------
            # 4. Coordinate Transformation
            # ---------------------------------------------------------
            # MediaPipe returns normalized coordinates (0.0 - 1.0) relative to the CROP.
            # We must convert them to Global Pixel Coordinates for logic,
            # and Global Normalized Coordinates for drawing.
            
            for lm_id, name in self.LANDMARK_NAMES.items():
                landmark = results.pose_landmarks.landmark[lm_id]
                
                # --- Step A: Local Normalized -> Local Pixel ---
                lx_px = landmark.x * crop_w
                ly_px = landmark.y * crop_h
                
                # --- Step B: Local Pixel -> Global Pixel ---
                gx_px = lx_px + crop_x1
                gy_px = ly_px + crop_y1
                
                # Use Global Pixel Coords for Dictionary (Logic)
                landmarks_dict[name] = [int(gx_px), int(gy_px)]
                
                # --- Step C: Global Pixel -> Global Normalized (For Drawing) ---
                # We update the landmark object in-place so mp_drawing works on full frame
                landmark.x = gx_px / w_frame
                landmark.y = gy_px / h_frame
                
                # Z coordinate: MediaPipe Z is roughly scale-relative. 
                # It might not need transformation if it's depth, 
                # but technically it depends on image width ratio if normalized.
                # Taking a simple approach: scale it by the ratio of widths
                landmark.z = landmark.z * (crop_w / w_frame) 

        return landmarks_dict, results

    def draw_landmarks(self, frame, results):
        """Helper to draw landmarks."""
        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
