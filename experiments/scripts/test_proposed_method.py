import cv2
import time
import numpy as np
import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from datetime import datetime


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import MediaPipe Estimator
from src.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
from src.vbt_analyzer import VBTAnalyzer

# Config
VIDEO_ROOT = os.path.join("src", "video")
RESULTS_ROOT = os.path.join("experiments", "results", "proposed_only")

# Set Japanese Font
plt.rcParams['font.family'] = 'MS Gothic'

def process_video_proposed(video_path, output_folder, args):
    filename = os.path.basename(video_path)
    print(f"Testing: {filename}...")
    
    # Initialize Proposed Method
    # Using 0.3 confidence as per previous settings for robustness
    estimator = Yolo11MediaPipeEstimator(
        model_path="yolo11n.pt", 
        min_detection_confidence=0.3, 
        min_tracking_confidence=0.3,
        margin_ratio=args.margin
    )
    analyzer = VBTAnalyzer(
        smoothing_window=args.smooth,
        velocity_threshold=args.threshold,
        velocity_threshold=args.threshold,
        filter_type=getattr(args, 'filter_type', 'average'),
        grip_finger=getattr(args, 'grip_finger', 'middle')
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Rotation Logic Removed as requested
    ROTATE = False

    start_time = time.time()
    frame_count = 0
    
    times = []
    velocities = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Rotation removed
        # if ROTATE:
        #    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        frame_count += 1
        curr_time = frame_count / fps
        
        # Process
        landmarks, pose_results = estimator.process_frame(frame)
        
        # VBT Logic
        vel = 0.0
        if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            # Use Robust Calibration Logic (Moved to VBTAnalyzer)
            if analyzer.calibration_factor is None:
                analyzer.attempt_robust_calibration(landmarks, pose_results, estimator.mp_pose)
            
            if analyzer.calibration_factor is not None:
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

    # Calculate Average Mean Velocity (Mean of Means)
    avg_mean_vel = 0.0
    if analyzer.rep_mean_velocities:
        avg_mean_vel = sum(analyzer.rep_mean_velocities) / len(analyzer.rep_mean_velocities)

    # Graph
    plt.figure(figsize=(10, 5))
    plt.plot(times, velocities, label=f'Proposed (MediaPipe) - {analyzer.rep_count} Reps', color='#007acc')
    plt.title(f"提案手法テスト: {filename} (Jitter: {jitter_score:.4f})", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    
    # Add Text Box for Stats
    stats_text = f"Mean Conc. Vel: {avg_mean_vel:.2f} m/s\nJitter: {jitter_score:.4f}"
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    graph_path = os.path.join(output_folder, f"test_{filename.replace('.mp4', '')}.png")
    plt.savefig(graph_path)
    plt.close()
    
    # Export detailed Rep Data
    rep_data = []
    for i in range(len(analyzer.rep_velocities)):
        p = analyzer.rep_velocities[i]
        m = 0.0
        if i < len(analyzer.rep_mean_velocities):
            m = analyzer.rep_mean_velocities[i]
        rep_data.append({"Rep": i+1, "PeakVel": round(p, 3), "MeanVel": round(m, 3)})
    
    rep_csv_path = os.path.join(output_folder, f"reps_{filename.replace('.mp4', '')}.csv")
    if rep_data:
        pd.DataFrame(rep_data).to_csv(rep_csv_path, index=False)

    return {
        "Video": filename,
        "Reps": analyzer.rep_count,
        "MeanVel": round(avg_mean_vel, 3), # Session Average
        "FPS": round(proc_fps, 1),
        "Jitter": round(jitter_score, 4),
        "Graph": graph_path,
        "RepDetail": rep_csv_path
    }

def main():
    parser = argparse.ArgumentParser(description="Run VBT Accuracy Test")
    parser.add_argument("--margin", type=float, default=0.3, help="ROI Margin Ratio (default: 0.3)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing Window (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Velocity Threshold (default: 0.05)")
    parser.add_argument("--filter_type", type=str, default="average", help="Filter (average, kalman, butterworth)")
    parser.add_argument("--grip_finger", type=str, default="middle", help="Grip finger on 81cm line")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include params in folder name for easy tracking
    param_str = f"m{int(args.margin*100)}_s{args.smooth}_t{int(args.threshold*100)}"
    out_dir = os.path.join(RESULTS_ROOT, f"run_{timestamp}_{param_str}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Starting Proposed Method Test ---")
    print(f"Config: Margin={args.margin}, Smooth={args.smooth}, Threshold={args.threshold}")
    print(f"Results will be saved to: {out_dir}")
    
    all_videos = glob.glob(os.path.join(VIDEO_ROOT, "**/*.mp4"), recursive=True)
    target_videos = [v for v in all_videos if "cropped" not in v and "demo" not in v]
    
    print(f"Found {len(target_videos)} videos.")
    
    results = []
    for v in target_videos:
        try:
            res = process_video_proposed(v, out_dir, args)
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
