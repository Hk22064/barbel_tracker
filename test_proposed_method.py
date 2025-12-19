import cv2
import time
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import MediaPipe Estimator
from VBT_mediapipe.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
from VBT_mediapipe.vbt_analyzer import VBTAnalyzer

# Config
VIDEO_ROOT = "VBT_mediapipe/video"
RESULTS_ROOT = "experiment_results/proposed_only"

# Set Japanese Font
plt.rcParams['font.family'] = 'MS Gothic'

def process_video_proposed(video_path, output_folder):
    filename = os.path.basename(video_path)
    print(f"Testing: {filename}...")
    
    # Initialize Proposed Method
    # Using 0.3 confidence as per previous settings for robustness
    estimator = Yolo11MediaPipeEstimator(model_path="yolo11n.pt", min_detection_confidence=0.3, min_tracking_confidence=0.3)
    analyzer = VBTAnalyzer(smoothing_window=5)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Rotation Logic (Re-enabled for Accuracy)
    # Without rotation, Pose Estimation fails on sideways videos like front_10rep.
    # Namun! 'front_5rep' works better WITHOUT rotation (based on app_yolo11 test).
    ROTATE = False
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    is_vertical = cap_w > cap_h
    is_10rep = "10rep" in filename
    is_vertical_file = "vertical" in video_path.lower()
    
    # Only rotate front_10rep (which is sideways and needs it)
    # Exclude front_5rep if it works better without.
    if is_vertical and (is_10rep or (is_vertical_file and "5rep" not in filename)):
         ROTATE = True
         print("  -> Applying 90deg Rotation")

    start_time = time.time()
    frame_count = 0
    
    times = []
    velocities = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if ROTATE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        frame_count += 1
        curr_time = frame_count / fps
        
        # Process
        landmarks, _ = estimator.process_frame(frame)
        
        # VBT Logic
        vel = 0.0
        if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            if analyzer.calibration_factor is None:
                # Robust Calibration Logic
                # 1. Try Wrist Distance (Standard)
                # 2. If Wrists overlap (Side View), fallback to Torso Length
                
                lw_x, lw_y = lw
                rw_x, rw_y = rw
                wrist_dist = np.sqrt((lw_x - rw_x)**2 + (lw_y - rw_y)**2)
                
                # Threshold: If wrist distance is too small (< 100px), assume Side View
                if wrist_dist > 100:
                    analyzer.calibrate(lw, rw)
                else:
                    # Fallback: Torso Calibration
                    # Assume Torso Length (Shoulder to Hip) is approx 50cm (0.5m)
                    if 'left_shoulder' in landmarks and 'left_hip' in landmarks:
                        ls = landmarks['left_shoulder']
                        lh = landmarks['left_hip']
                        torso_dist = np.sqrt((ls[0] - lh[0])**2 + (ls[1] - lh[1])**2)
                        
                        if torso_dist > 0:
                            analyzer.calibration_factor = 0.50 / torso_dist
                            print(f"[Calibration] Side View Detected (WristDist: {wrist_dist:.1f}px). Using Torso (Factor: {analyzer.calibration_factor:.4f} m/px)")
                        else:
                            # Last resort: Fixed conservative factor
                             analyzer.calibration_factor = 0.0015
                    else:
                         analyzer.calibration_factor = 0.0015
                vel = analyzer.calculate_velocity(bar_y, curr_time)
                analyzer.process_rep(vel)
        
        velocities.append(vel)
        times.append(curr_time)
        
    cap.release()
    estimator.close()
    
    elapsed = time.time() - start_time
    proc_fps = frame_count / elapsed if elapsed > 0 else 0
    
    # Calculate Jitter Score (Mean Absolute Velocity Change)
    # Lower is smoother.
    # We ignore the first few frames where velocity might be unstable.
    if len(velocities) > 10:
        raw_diffs = np.abs(np.diff(velocities[5:])) # Skip first 5
        jitter_score = np.mean(raw_diffs)
    else:
        jitter_score = 0.0

    # Graph
    plt.figure(figsize=(10, 5))
    plt.plot(times, velocities, label=f'Proposed (MediaPipe) - {analyzer.rep_count} Reps', color='#007acc')
    plt.title(f"提案手法テスト: {filename} (Jitter: {jitter_score:.4f})", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    graph_path = os.path.join(output_folder, f"test_{filename.replace('.mp4', '')}.png")
    plt.savefig(graph_path)
    plt.close()
    
    return {
        "Video": filename,
        "Reps": analyzer.rep_count,
        "FPS": round(proc_fps, 1),
        "Jitter": round(jitter_score, 4),
        "Graph": graph_path
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_ROOT, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Starting Proposed Method Test ---")
    print(f"Results will be saved to: {out_dir}")
    
    all_videos = glob.glob(os.path.join(VIDEO_ROOT, "**/*.mp4"), recursive=True)
    target_videos = [v for v in all_videos if "cropped" not in v and "demo" not in v]
    
    print(f"Found {len(target_videos)} videos.")
    
    results = []
    for v in target_videos:
        try:
            res = process_video_proposed(v, out_dir)
            results.append(res)
            print(f"  -> Reps: {res['Reps']}, FPS: {res['FPS']}")
        except Exception as e:
            print(f"  -> Error: {e}")
            
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(out_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print("\n--- Summary ---")
        print(df)
        print(f"\nSaved full summary to {csv_path}")

if __name__ == "__main__":
    main()
