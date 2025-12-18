import cv2
import time
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import VBT modules
from VBT_mediapipe.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
from VBT_mediapipe.vbt_analyzer import VBTAnalyzer
from VBT_mediapipe.comp_object_estimator import CompObjectEstimator

# Config
VIDEO_ROOT = "VBT_mediapipe/video"
RESULTS_ROOT = "experiment_results"

# Set Japanese Font
plt.rcParams['font.family'] = 'MS Gothic'

def process_single_video(video_path, output_folder):
    filename = os.path.basename(video_path)
    print(f"Processing: {filename}...")
    
    # --- A. Proposed Method (MediaPipe) ---
    print(f"  Running Proposed Method (MediaPipe)...")
    estimator = Yolo11MediaPipeEstimator(model_path="yolo11n.pt", min_detection_confidence=0.3, min_tracking_confidence=0.3)
    analyzer = VBTAnalyzer(smoothing_window=5)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_time = time.time()
    
    hybrid_velocity = []
    hybrid_time = []
    
    frame_count = 0
    # Rotation Check for Hybrid
    ROTATE = False
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cap_w > cap_h and ("10rep" in filename or "vertical" in video_path.lower()):
         ROTATE = True

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
        
        # Analyze
        vel = 0.0
        if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            if analyzer.calibration_factor is None:
                analyzer.calibrate(lw, rw)
            else:
                vel = analyzer.calculate_velocity(bar_y, curr_time)
                analyzer.process_rep(vel)
        
        hybrid_velocity.append(vel)
        hybrid_time.append(curr_time)

    cap.release()
    estimator.close()
    
    elapsed_hybrid = time.time() - start_time
    fps_hybrid = frame_count / elapsed_hybrid if elapsed_hybrid > 0 else 0
    reps_hybrid = analyzer.rep_count
    
    # --- B. Comparison Method (Object Detection + Plate Scale) ---
    print(f"  Running Comp Method (Object + PlateScale)...")
    comp_est = CompObjectEstimator(model_path="models/Model_B_Clump_n/train/weights/best.pt")
    reps_object, fps_object, object_time, object_velocity = comp_est.process_video_and_count(video_path)
    
    # --- Generate Graph (Japanese) ---
    plt.figure(figsize=(10, 5))
    
    # Plot Hybrid
    plt.plot(hybrid_time, hybrid_velocity, color='#007acc', label=f'提案手法 (Hybrid) - {reps_hybrid} 回', linewidth=2)
    
    # Plot Object (if available)
    if object_time and len(object_time) == len(object_velocity):
        # Trim to match length if needed (FPS diff?)
        # Just plot direct
        plt.plot(object_time, object_velocity, color='#ff7f0e', label=f'比較手法 (Object) - {reps_object} 回', linestyle='--', alpha=0.8)
    
    plt.title(f"挙上速度プロファイル比較: {filename}", fontsize=14, fontweight='bold')
    plt.xlabel("時間 (秒)", fontsize=12)
    plt.ylabel("速度 (m/s)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    
    graph_path = os.path.join(output_folder, f"graph_{filename.replace('.mp4', '')}.png")
    plt.savefig(graph_path)
    plt.close()
    
    return {
        "Video": filename,
        "Hybrid Reps": reps_hybrid,
        "Hybrid FPS": round(fps_hybrid, 1),
        "Object Reps": reps_object,
        "Object FPS": round(fps_object, 1),
        "Graph": graph_path
    }

def main():
    # 1. Setup Experiment Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_ROOT, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Starting Batch Analysis. Results: {exp_dir}")
    print("Graph titles will be in Japanese.")
    print("Comparison Method now uses 'analyze_scale' logic (Plate Detection).")
    
    # 2. Find Videos
    all_videos = glob.glob(os.path.join(VIDEO_ROOT, "**/*.mp4"), recursive=True)
    target_videos = [v for v in all_videos if "cropped" not in v and "demo" not in v]
    
    print(f"Found {len(target_videos)} videos.")
    
    results = []
    
    # 3. Process Loop
    for video_path in target_videos:
        try:
            res = process_single_video(video_path, exp_dir)
            results.append(res)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # 4. Save Summary
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(exp_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Batch Analysis Complete. Summary saved to {csv_path}")
        print(df[["Video", "Hybrid Reps", "Object Reps", "Hybrid FPS"]])
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
