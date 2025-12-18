import cv2
import argparse
import sys
import os

sys.path.append(os.getcwd())

from VBT_mediapipe.yolo_hybrid_estimator import YoloHybridEstimator
from VBT_mediapipe.vbt_analyzer import VBTAnalyzer

def main():
    video_path = "video/horizontal/front_9rep.mp4"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    print(f"Testing Hybrid Estimator on: {video_path}")
    
    # Use same model as user (default yolo11s now)
    estimator = YoloHybridEstimator(yolo_model="models/yolo11x.pt") 
    analyzer = VBTAnalyzer()
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        landmarks, _ = estimator.process_frame(frame)
        
        if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            # Auto-calib logic
            if analyzer.calibration_factor is None:
                # Force calibrate on first good frame for debugging
                analyzer.calibrate(lw, rw)
            else:
                vel = analyzer.calculate_velocity(bar_y, frame_count/30.0) # approx time
                analyzer.process_rep(vel)

    cap.release()
    print(f"Final Rep Count: {analyzer.rep_count}")
    print(f"Calibration Factor: {analyzer.calibration_factor}")

if __name__ == "__main__":
    main()
