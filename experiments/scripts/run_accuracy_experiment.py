
import cv2
import time
import numpy as np
import os
import sys
import pandas as pd

# Add project root to sys.path to allow imports
# Add project root to sys.path to allow imports
# This script is in experiments/scripts/ (2 levels deep from root)
# But absolute path is safer.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
from src.vbt_analyzer import VBTAnalyzer

# Config
VIDEO_ROOT = os.path.join("src", "video")
TRUTH_ROOT = os.path.join("experiments", "truth")
OUTPUT_DIR = os.path.join("experiments", "results", "final_accuracy_results")

# Target Definitions
# (Video Relative Path, Truth Filename)
TARGETS = [
    ("horizontal/front_9rep.mp4", "front9rep_目測平均速度.txt"),
    ("vertical/front_10rep.mp4", "front10rep_目測平均速度.txt"),
    ("vertical/front_5rep.mp4", "front5rep_目測平均速度.txt"),
]

def load_ground_truth(txt_path):
    """
    Parses txt files like:
    1rep 0.18 27
    2rep 0.19 21
    ...
    Returns list of dict: [{'Rep': 1, 'Vel': 0.18, 'Frames': 27}, ...]
    """
    data = []
    if not os.path.exists(txt_path):
        print(f"[WARN] GT File not found: {txt_path}")
        return data

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            # Expecting >= 3 parts: [RepStr, Vel, Frames]
            # e.g. "1rep", "0.18", "27"
            if len(parts) >= 3:
                try:
                    rep_idx = int(parts[0].replace('rep', '').replace(':', ''))
                    val = float(parts[1])
                    frames = int(parts[2])
                    data.append({
                        'Rep': rep_idx,
                        'ManualVel': val,
                        'ManualFrames': frames
                    })
                except ValueError:
                    continue
    return data

import argparse

def run_experiment():
    parser = argparse.ArgumentParser(description="Final Accuracy Experiment")
    parser.add_argument("--margin", type=float, default=0.3, help="ROI Margin Ratio")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing Window")
    parser.add_argument("--threshold", type=float, default=0.05, help="Velocity Threshold")
    parser.add_argument("--filter_type", type=str, default="average", help="Filter Strategy")
    parser.add_argument("--grip_finger", type=str, default="middle", help="Grip finger on 81cm line")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Starting Final Accuracy Experiment ---")
    print(f"Config: Margin={args.margin}, Smooth={args.smooth}, Threshold={args.threshold}, Filter={args.filter_type}")
    
    for video_rel, truth_file in TARGETS:
        video_path = os.path.join(VIDEO_ROOT, video_rel)
        truth_path = os.path.join(TRUTH_ROOT, truth_file)
        
        if not os.path.exists(video_path):
             print(f"[SKIP] Video not found: {video_path}")
             continue
             
        print(f"\nProcessing: {video_rel}")
        
        # Init System
        estimator = Yolo11MediaPipeEstimator(model_path="yolo11n.pt", margin_ratio=args.margin)
        analyzer = VBTAnalyzer(smoothing_window=args.smooth, velocity_threshold=args.threshold, filter_type=args.filter_type, grip_finger=args.grip_finger)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_idx = 0
        
        # Tracking Rep Duration
        concentric_start_frame = None
        system_rep_frames = [] # List of frame counts for each committed rep
        
        # We need to detect when a rep is committed to store the duration.
        # VBTAnalyzer.rep_count increments on commit.
        last_rep_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # --- Processing ---
            landmarks, results = estimator.process_frame(frame)
            
            bar_y = 0.0
            if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
                lw = landmarks['left_wrist']
                rw = landmarks['right_wrist']
                bar_y = (lw[1] + rw[1]) / 2.0
                
                if analyzer.calibration_factor is None:
                    analyzer.attempt_robust_calibration(landmarks, results, estimator.mp_pose)
                
                if analyzer.calibration_factor is not None:
                    vel = analyzer.calculate_velocity(bar_y, current_time)
                    analyzer.process_rep(vel)
            
            # --- Duration Tracking ---
            state = analyzer.current_state
            
            # Start counting when we enter CONCENTRIC
            if state == analyzer.STATE_CONCENTRIC:
                if concentric_start_frame is None:
                    concentric_start_frame = frame_idx
            else:
                pass

            # Check for Rep Commit
            if analyzer.rep_count > last_rep_count:
                if concentric_start_frame is not None:
                    duration = frame_idx - concentric_start_frame
                    system_rep_frames.append(duration)
                else:
                    system_rep_frames.append(0) # Fallback
                
                concentric_start_frame = None
                last_rep_count = analyzer.rep_count
            
            # Fail-safe reset if we go back to ECCENTRIC without committing
            if state == analyzer.STATE_ECCENTRIC:
                concentric_start_frame = None
                
        cap.release()
        estimator.close()
        
        # --- Analysis vs Truth ---
        gt_data = load_ground_truth(truth_path)
        
        # Combine into DataFrame
        report_rows = []
        
        max_reps = max(len(gt_data), len(analyzer.rep_mean_velocities))
        
        total_vel_error = 0.0
        total_frame_error = 0
        valid_pairs = 0
        
        for i in range(max_reps):
            rep_no = i + 1
            
            # GT
            gt_vel = None
            gt_frames = None
            if i < len(gt_data):
                gt_vel = gt_data[i]['ManualVel']
                gt_frames = gt_data[i]['ManualFrames']
            
            # System
            sys_vel = None
            sys_frames = None
            if i < len(analyzer.rep_mean_velocities):
                sys_vel = analyzer.rep_mean_velocities[i]
                sys_frames = system_rep_frames[i] if i < len(system_rep_frames) else None
            
            # Errors
            err_vel = None
            err_vel_pct = None
            diff_frames = None
            
            if gt_vel is not None and sys_vel is not None:
                err_vel = abs(sys_vel - gt_vel)
                if gt_vel > 0:
                    err_vel_pct = (err_vel / gt_vel) * 100
                total_vel_error += err_vel
                valid_pairs += 1
                
            if gt_frames is not None and sys_frames is not None:
                diff_frames = sys_frames - gt_frames
                total_frame_error += abs(diff_frames)
                
            row = {
                "Rep": rep_no,
                "Manual Vel": gt_vel,
                "System Vel": round(sys_vel, 3) if sys_vel else None,
                "Diff Vel": round(err_vel, 3) if err_vel else None,
                "Diff %": round(err_vel_pct, 1) if err_vel_pct is not None else None,
                "Manual Frames": gt_frames,
                "System Frames": sys_frames,
                "Diff Frames": diff_frames
            }
            report_rows.append(row)
            
        # Save CSV
        df = pd.DataFrame(report_rows)
        # Add Filter type and params to filename
        base_name = os.path.basename(video_rel).replace('.mp4', '')
        param_str = f"s{args.smooth}_th{int(args.threshold*100)}"
        out_name = f"accuracy_{base_name}_{args.filter_type}_{param_str}.csv"
        csv_path = os.path.join(OUTPUT_DIR, out_name)
        df.to_csv(csv_path, index=False)
        
        # Console Output
        print(f"  -> Truth Loaded: {len(gt_data)} reps")
        print(f"  -> System Detected: {analyzer.rep_count} reps")
        if valid_pairs > 0:
            avg_vel_err = total_vel_error / valid_pairs
            avg_frame_err = total_frame_error / valid_pairs
            print(f"  -> Avg Velocity Error: {avg_vel_err:.4f} m/s")
            print(f"  -> Avg Frame Diff: {avg_frame_err:.1f} frames")
        
        print("\n" + df.to_string())
        print(f"Saved to: {csv_path}")
        print("-" * 60)

if __name__ == "__main__":
    run_experiment()
