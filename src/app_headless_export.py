import cv2
import time
import argparse
import numpy as np
import os
import sys

# Add current dir to path to sure imports work
sys.path.append(os.getcwd())

from pose_estimator import PoseEstimator
from vbt_analyzer import VBTAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("output_file", type=str, help="Path to output .npy file")
    args = parser.parse_args()

    # Initialize Modules
    pose_estimator = PoseEstimator()
    vbt_analyzer = VBTAnalyzer(smoothing_window=5)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    velocity_history = []
    
    frame_idx = 0
    
    print("Running MediaPipe Analysis...")
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Use Video Time instead of System Time
        current_time = frame_idx / fps
        
        # 1. Pose Estimation
        landmarks_dict, results = pose_estimator.process_frame(frame)
        
        bar_y = 0
        velocity = 0.0
        
        if 'left_wrist' in landmarks_dict and 'right_wrist' in landmarks_dict:
            lw = landmarks_dict['left_wrist']
            rw = landmarks_dict['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            # Force Calibration for test purposes if not set?
            # The original app auto-calibrates or manually calibrates.
            # "The user says 81cm".
            # In the user's `app.py` usage instructions, it says press 'c'.
            # Since this is headless, I should FORCE calibration or attempt auto-cal immediately.
            # But auto-cal needs "stillness".
            # For this comparison to be fair with YOLO models (which use 45cm plate), 
            # we need to be careful. MediaPipe logic uses "Wrist Distance = 81cm" (Grip width).
            # This is a different reference!
            # If I want to compare standard MediaPipe VBT logic, I should use its own logic.
            # But the scale will be different (Grip based vs Plate based).
            # To compare *Trajectory shape* and *Timing*, scale doesn't matter much (just amplitude).
            # But the user asked for "velocity graph comparison".
            # If scales are different, curves won't overlap.
            # Assumption: User wants to see "How MediaPipe performs".
            # I will attempt auto-calibration just like the app.
            
            if vbt_analyzer.calibration_factor is None:
                 # Hack: Force calibration valid on first frame using current wrist dist
                 # This mimics "User configured it perfectly".
                 # Using 81cm (0.81m) as standard grip width.
                 vbt_analyzer.calibrate(lw, rw, real_distance_m=0.81)
                 velocity = 0.0
            else:
                 velocity = vbt_analyzer.calculate_velocity(bar_y, current_time)
                 vbt_analyzer.process_rep(velocity)
        
        velocity_history.append(velocity)
        frame_idx += 1

    cap.release()
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total Time: {elapsed:.2f} s")
    print(f"Average FPS: {frame_idx / elapsed:.2f}")

    # Save Data
    np.save(args.output_file, np.array(velocity_history))
    print(f"MediaPipe analysis complete. Frames: {len(velocity_history)}")
    print(f"Saved velocity data to {args.output_file}")
    print(f"Total Reps Detected: {vbt_analyzer.rep_count}")

if __name__ == "__main__":
    main()
