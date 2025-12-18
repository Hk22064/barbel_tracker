import cv2
import numpy as np
from ultralytics import YOLO

class YoloPoseHybridEstimator:
    def __init__(self, pose_model_path="yolo11x-pose.pt", detect_model_path="yolo11n.pt", device=None):
        """
        Initializes the YoloPoseHybridEstimator.
        Uses YOLO11n for Initial Detection (Lock) and YOLO11x-Pose for Detail.
        device: 'cpu' or 'cuda'
        """
        print(f"Loading YOLO Detect model: {detect_model_path} (Device: {device})...")
        self.detect_model = YOLO(detect_model_path)
        
        print(f"Loading YOLO Pose model: {pose_model_path} (Device: {device})...")
        self.pose_model = YOLO(pose_model_path)
        
        self.device = device
        
        # Lock State
        self.locked_box = None
        self.frame_count = 0
        self.scan_interval = 30 # Check every ~1 second
        self.prev_scan_box = None
        self.current_crop_box = None
        
        # Stability Config
        self.stability_threshold = 50

    def process_frame(self, frame):
        """
        Process a single frame.
        """
        landmarks_dict = {}
        self.frame_count += 1
        h_img, w_img, _ = frame.shape
        
        # 1. Determine Crop Box (Stable Lock Logic)
        if self.locked_box is not None:
             self.current_crop_box = self.locked_box
             
        elif self.current_crop_box is None or self.frame_count % self.scan_interval == 0:
            # Run YOLO Detect
            results = self.detect_model(frame, classes=[0], verbose=False, device=self.device)
            target_box = None
            max_area = 0
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    area = w * h
                    if area > max_area:
                        max_area = area
                        target_box = xyxy
            
            if target_box is not None:
                x1, y1, x2, y2 = target_box
                box_w = x2 - x1
                box_h = y2 - y1
                
                margin_x = int(box_w * 0.2)
                margin_y = int(box_h * 0.2)
                
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(w_img, x2 + margin_x)
                crop_y2 = min(h_img, y2 + margin_y)
                new_crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
                
                # Check Stability
                if self.prev_scan_box is not None:
                    p_x1, p_y1, p_x2, p_y2 = self.prev_scan_box
                    n_x1, n_y1, n_x2, n_y2 = new_crop_box
                    diff = abs(p_x1 - n_x1) + abs(p_y1 - n_y1) + abs(p_x2 - n_x2) + abs(p_y2 - n_y2)
                    
                    if diff < self.stability_threshold:
                        self.locked_box = new_crop_box
                        print(f"[YoloPose] Stable Lock Engaged! Diff: {diff}")
                    else:
                        print(f"[YoloPose] Unstable. Diff: {diff}")
                        
                self.prev_scan_box = new_crop_box
                self.current_crop_box = new_crop_box
            else:
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

        # 3. YOLO Pose Estimation on Crop
        pose_results = self.pose_model(cropped_frame, verbose=False, device=self.device)
        
        # 4. Coordinate Transformation
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            # YOLO Keypoints are (N, 17, 3) -> [x, y, conf]
            # We assume single person in crop (or take first)
            kpts = pose_results[0].keypoints.xy.cpu().numpy() # (N, 17, 2)
            if len(kpts) > 0:
                person_kpts = kpts[0] # Take first person
                
                # COCO Keypoint Mapping
                coco_map = {
                    9: 'left_wrist',
                    10: 'right_wrist',
                    5: 'left_shoulder',
                    6: 'right_shoulder',
                    7: 'left_elbow',
                    8: 'right_elbow',
                    11: 'left_hip',
                    12: 'right_hip',
                    0: 'nose'
                }
                
                for idx, name in coco_map.items():
                    if idx < len(person_kpts):
                        lx, ly = person_kpts[idx]
                        if lx == 0 and ly == 0: continue # Invalid
                        
                        # Global Transform
                        gx = crop_x1 + lx
                        gy = crop_y1 + ly
                        
                        landmarks_dict[name] = [int(gx), int(gy)]
                        
        landmarks_dict['bbox'] = self.current_crop_box
        return landmarks_dict, None

    def reset(self):
        print("[YoloPose] Resetting Target Lock...")
        self.locked_box = None
        self.current_crop_box = None
        self.prev_scan_box = None
        self.frame_count = 0

    def close(self):
        pass 
