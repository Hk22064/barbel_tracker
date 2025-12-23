import cv2
import numpy as np
import os
import sys
import argparse
import pandas as pd
import glob
import traceback
from VBT_mediapipe.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
from VBT_mediapipe.vbt_analyzer import VBTAnalyzer

# --- Configuration ---
VIDEO_ROOT = r"C:\Users\kurau\Bench_pose\new_mylab\barbel_tracker\VBT_mediapipe\video"
GT_DIR = r"C:\Users\kurau\Bench_pose\new_mylab\barbel_tracker\VBT_experiment"
DEBUG_CSV = "optimization_debug.csv"
RESULTS_CSV = "optimization_results_average.csv"

# Target Videos and Ground Truth Rep Counts
TARGETS = {
    "front_5rep": {
        "rel_path": r"vertical\front_5rep.mp4", 
        "gt_reps": 5, 
        "gt_txt": "front5rep_目測平均速度.txt"
    },
    "front_9rep": {
        "rel_path": r"horizontal\front_9rep.mp4", 
        "gt_reps": 9, 
        "gt_txt": "front9rep_目測平均速度.txt"
    },
    "front_10rep": {
        "rel_path": r"vertical\front_10rep.mp4", 
        "gt_reps": 10, 
        "gt_txt": "front10rep_目測平均速度.txt"
    }
}

# FULL PARAM GRID
PARAMS = {
    "filter_type": ["average"],
    "smooth": [3, 5, 7, 9],
    "threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
}

def load_ground_truth(txt_path):
    try:
        if not os.path.exists(txt_path):
            return []
        velocities = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        velocities.append(float(parts[1]))
                    except ValueError:
                        continue
        return velocities
    except Exception as e:
        print(f"Error loading GT {txt_path}: {e}")
        return []

def log_crash(msg):
    with open("crash_log_v2.txt", "a") as f:
        f.write(msg + "\n")

def run_analysis(video_path, smooth, threshold, filter_type):
    try:
        if not os.path.exists(video_path):
            log_crash(f"Video not found: {video_path}")
            return None, []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_crash(f"Could not open video: {video_path}")
            return None, []

        estimator = Yolo11MediaPipeEstimator(model_path="yolo11n.pt", margin_ratio=0.3)
        # FIX: 'smoothing_window' instead of 'smooth_window'
        analyzer = VBTAnalyzer(
            velocity_threshold=threshold, 
            smoothing_window=smooth, 
            filter_type=filter_type
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, results = estimator.process_frame(frame)

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
                lw = landmarks['left_wrist']
                rw = landmarks['right_wrist']
                bar_y = (lw[1] + rw[1]) / 2.0
                
                if analyzer.calibration_factor is None:
                    analyzer.attempt_robust_calibration(landmarks, results, estimator.mp_pose)
                
                if analyzer.calibration_factor is not None:
                    vel = analyzer.calculate_velocity(bar_y, current_time)
                    analyzer.process_rep(vel)

        cap.release()
        estimator.close()
        
        # Construct simplified data structure expected by evaluator
        reps_data = [{'avg_velocity': v} for v in analyzer.rep_mean_velocities]
        return analyzer.rep_count, reps_data
        
    except Exception as e:
        # Restore normal error logging
        import traceback
        err_msg = f"CRASH in run_analysis (Smooth={smooth}, Th={threshold}): {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(err_msg)
        log_crash(err_msg)
        return None, []

def evaluate_and_log(smooth, threshold, filter_type):
    set_detected_reps = {}
    set_reps_data = {}
    mismatch = False
    
    # Check Rep Counts
    for key, info in TARGETS.items():
        video_path = os.path.join(VIDEO_ROOT, info["rel_path"])
        gt_reps = info["gt_reps"]
        
        res = run_analysis(video_path, smooth, threshold, filter_type)
        if res is None or res[0] is None:
             rep_count = -1
             data = []
        else:
             rep_count, data = res
             
        set_detected_reps[key] = rep_count
        set_reps_data[key] = data
        
        status = "MATCH" if rep_count == gt_reps else "MISMATCH"
        print(f"  > {video_path} -> Reps: {rep_count} / {gt_reps} [{status}]")
        
        # Log immediately
        log_entry = {
            "filter": filter_type,
            "smooth": smooth,
            "threshold": threshold,
            "video": key,
            "detected_reps": rep_count,
            "gt_reps": gt_reps,
            "status": status
        }
        # Append to CSV
        df = pd.DataFrame([log_entry])
        hdr = not os.path.exists(DEBUG_CSV)
        df.to_csv(DEBUG_CSV, mode='a', header=hdr, index=False)
        
        if rep_count != gt_reps:
            mismatch = True
            # Break early
            return None 

    # Calculate Error
    total_error = 0
    valid_count = 0
    
    for key, info in TARGETS.items():
        reps_data = set_reps_data[key]
        gt_txt_path = os.path.join(GT_DIR, info["gt_txt"])
        manual_velocities = load_ground_truth(gt_txt_path)
        
        n = min(len(reps_data), len(manual_velocities))
        for i in range(n):
            sys_vel = reps_data[i]['avg_velocity']
            man_vel = manual_velocities[i]
            total_error += abs(sys_vel - man_vel)
            valid_count += 1
            
    if valid_count == 0:
        return None
        
    avg_error = total_error / valid_count
    return avg_error

def main():
    # Remove old debug log
    if os.path.exists(DEBUG_CSV):
        try: os.remove(DEBUG_CSV)
        except: pass

    results = []
    
    print(f"Starting Optimization (Full Mode)...")
    print(f"Filter: {PARAMS['filter_type']}")
    print(f"Smooth: {PARAMS['smooth']}")
    print(f"Threshold: {PARAMS['threshold']}")
    
    total_combos = len(PARAMS['smooth']) * len(PARAMS['threshold'])
    count = 0

    for s in PARAMS['smooth']:
        for th in PARAMS['threshold']:
            count += 1
            print(f"[{count}/{total_combos}] Smooth={s}, Th={th} ...")
            sys.stdout.flush()
            
            try:
                mae = evaluate_and_log(s, th, "average")
                
                if mae is None:
                    print("  -> FAIL (Mismatch)")
                else:
                    print(f"  -> PASS | MAE: {mae:.4f}")
                    results.append({
                        "filter": "average",
                        "smooth": s,
                        "threshold": th,
                        "mae": mae
                    })
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()

    print("-" * 50)
    print("Optimization Complete.")
    
    if not results:
        print("No valid parameters found.")
        return

    sorted_results = sorted(results, key=lambda x: x['mae'])
    best = sorted_results[0]
    
    print("\n🏆 BEST PARAMETERS 🏆")
    print(f"Smooth: {best['smooth']}")
    print(f"Threshold: {best['threshold']}")
    print(f"MAE: {best['mae']:.4f}")
    
    df = pd.DataFrame(sorted_results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved best results to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
