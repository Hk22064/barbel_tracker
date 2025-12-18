"""
==============================================================================
VBT Hybrid Analyzer (YOLO Detect + MediaPipe Pose)
==============================================================================

【概要 / Overview】
YOLOv8で人物を検出し、その領域をクロップしてからMediaPipeで姿勢推定を行う
ハイブリッド構成のVBTアプリです。
遠距離からの撮影や背景が複雑な場合でも、YOLOの検出力でロバスト性を高めます。

【実行手順 / Execution Instructions】
1. Conda環境の有効化:
   conda activate yolo

2. 実行:
   - Webカメラ:
     python VBT_mediapipe/app_hybrid.py
   - 動画ファイル:
     python VBT_mediapipe/app_hybrid.py --video data/squat.mp4

==============================================================================
"""

import cv2
import time
import argparse
import numpy as np
import sys
import os

# Add current dir to path to sure imports work
# Assumes running from 'barbel_tracker' root usually, but VBT_mediapipe package structure applies
sys.path.append(os.getcwd())

# Import Hybrid Estimator
from VBT_mediapipe.yolo_hybrid_estimator import YoloHybridEstimator
from VBT_mediapipe.vbt_analyzer import VBTAnalyzer

def draw_text(img, text, pos, color=(0, 255, 0), font_scale=0.7, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_graph(image, velocity_history, graph_x, graph_y, graph_w, graph_h, max_val=2.0):
    cv2.rectangle(image, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), -1)
    if len(velocity_history) < 2: return
    points = []
    for i, val in enumerate(velocity_history):
        val = max(0, min(val, max_val))
        x = graph_x + int((i / len(velocity_history)) * graph_w)
        y = graph_y + graph_h - int((val / max_val) * graph_h)
        points.append((x, y))
    if points:
        cv2.polylines(image, [np.array(points)], isClosed=False, color=(0, 255, 255), thickness=1)

def get_color_by_fatigue(drop_off_percent):
    if drop_off_percent < 10: return (0, 255, 0)
    elif drop_off_percent < 20: return (0, 255, 255)
    else: return (0, 0, 255)

def main():
    parser = argparse.ArgumentParser(description="VBT Hybrid Analyzer")
    parser.add_argument("--video", type=str, help="Path to video file", default=None)
    parser.add_argument("--model", type=str, default="models/yolo11s.pt", help="YOLO model path")
    args = parser.parse_args()

    # Initialize Hybrid Estimator
    print("Initializing YoloHybridEstimator...")
    pose_estimator = YoloHybridEstimator(yolo_model=args.model)
    vbt_analyzer = VBTAnalyzer(smoothing_window=5)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Opening video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Opening Webcam")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    graph_history = []
    max_graph_points = 200

    while True:
        ret, frame = cap.read()
        if not ret: break

        # ----------------------------------------------------
        # Display Resize Logic (Same as updated app.py)
        # ----------------------------------------------------
        display_frame = frame.copy() # Make a copy for display
        max_h = 800
        max_w = 1280
        h, w = frame.shape[:2]
        
        scale_disp = 1.0
        if h > max_h or w > max_w:
            scale_h = max_h / h
            scale_w = max_w / w
            scale_disp = min(scale_h, scale_w)
            new_w = int(w * scale_disp)
            new_h = int(h * scale_disp)
            # Use resized frame for display, but original 'frame' for processing
            display_frame = cv2.resize(frame, (new_w, new_h))
        else:
            display_frame = frame

        current_time = time.time()
        
        # ----------------------------------------------------
        # 1. Hybrid Processing (Detect -> Crop -> Pose)
        # ----------------------------------------------------
        # process_frame now handles the global Coordinate Transformation
        landmarks_dict, results = pose_estimator.process_frame(frame)
        
        # Draw on display frame
        # Since 'results' contains normalized coords 0.0-1.0, 
        # draw_landmarks will draw on 'display_frame' correctly regardless of resolution
        pose_estimator.draw_landmarks(display_frame, results)

        # ----------------------------------------------------
        # 2. VBT Logic (Same as standard app)
        # ----------------------------------------------------
        bar_y = 0
        velocity = 0.0
        
        if 'left_wrist' in landmarks_dict and 'right_wrist' in landmarks_dict:
            lw = landmarks_dict['left_wrist']
            rw = landmarks_dict['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            if vbt_analyzer.calibration_factor is None:
                if args.video:
                    vbt_analyzer.calibrate(lw, rw)
                    velocity = 0.0
                else:
                    raw_vel = 0.0
                    if vbt_analyzer.prev_y is not None:
                         raw_vel = abs(vbt_analyzer.prev_y - bar_y)
                    vbt_analyzer.attempt_auto_calibration(lw, rw, raw_vel)
                    vbt_analyzer.calculate_velocity(bar_y, current_time)
            else:
                velocity = vbt_analyzer.calculate_velocity(bar_y, current_time)
                vbt_analyzer.process_rep(velocity)

        # ----------------------------------------------------
        # 3. Update Graph & UI
        # ----------------------------------------------------
        graph_history.append(velocity)
        if len(graph_history) > max_graph_points: graph_history.pop(0)

        # Drawing UI on 'display_frame'
        ui_y = 30
        
        # Calibration Status
        if vbt_analyzer.calibration_factor:
            cal_text = "Calibration: OK (Hybrid)"
            cal_color = (0, 255, 0)
        else:
            cal_text = "Calibrating..."
            cal_color = (0, 255, 255)
        draw_text(display_frame, cal_text, (10, ui_y), color=cal_color)
        ui_y += 40
        
        # Rep Count
        rep_text = f"Reps: {vbt_analyzer.rep_count}  [{vbt_analyzer.current_state}]"
        draw_text(display_frame, rep_text, (10, ui_y), (0, 255, 0))
        ui_y += 40
        
        # Velocity
        if vbt_analyzer.calibration_factor:
            fatigue = vbt_analyzer.get_fatigue_status()
            col = get_color_by_fatigue(fatigue)
            draw_text(display_frame, f"Vel: {velocity:.2f} m/s", (10, ui_y), col)
            ui_y += 35
            if vbt_analyzer.rep_velocities:
                 draw_text(display_frame, f"Peak: {vbt_analyzer.rep_velocities[-1]:.2f} m/s", (10, ui_y), col)

        # Graph
        d_h, d_w, _ = display_frame.shape
        draw_graph(display_frame, graph_history, 10, d_h - 160, 300, 150)

        cv2.imshow('VBT Hybrid Analyzer', display_frame)

        # Event Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        # Check if window is closed by user (X button)
        if cv2.getWindowProperty('VBT Hybrid Analyzer', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
