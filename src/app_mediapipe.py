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

# Add project root to sys.path to allow imports from src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MediaPipe Estimator
try:
    from src.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
    from src.vbt_analyzer import VBTAnalyzer
except ImportError:
    # Fallback if running from within src without package context
    from yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
    from vbt_analyzer import VBTAnalyzer

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
    parser = argparse.ArgumentParser(description="VBT MediaPipe Analyzer")
    parser.add_argument("--video", type=str, help="Path to video file", default=None)
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path")
    
    # Tuning Parameters
    parser.add_argument("--margin", type=float, default=0.3, help="ROI Margin Ratio (default: 0.3)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing Window (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Velocity Threshold (default: 0.05)")
    parser.add_argument("--filter_type", type=str, default="average", help="Filter Type (average, kalman, butterworth)")
    parser.add_argument("--grip_finger", type=str, default="middle", help="Grip finger on 81cm line (index, middle, ring, pinky)")
    args = parser.parse_args()

    # Initialize MediaPipe Estimator
    print(f"Initializing Yolo11MediaPipeEstimator (Margin={args.margin})...")
    pose_estimator = Yolo11MediaPipeEstimator(model_path=args.model, margin_ratio=args.margin)
    
    print(f"Initializing VBTAnalyzer (Smooth={args.smooth}, Threshold={args.threshold}, Filter={args.filter_type}, Grip={args.grip_finger})...")
    vbt_analyzer = VBTAnalyzer(smoothing_window=args.smooth, velocity_threshold=args.threshold, filter_type=args.filter_type, grip_finger=args.grip_finger)

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
    paused = False # Pause State

    force_one_frame = False

    while True:
        if not paused or force_one_frame:
            ret, frame = cap.read()
            if not ret: break
            force_one_frame = False
        
        # If paused and no force read, 'frame' holds the last image.
        # We continue to process/draw it (pointless for static, but keeps UI responsive).

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

        if args.video:
            # Use Video Timestamp for accurate VBT analysis regardless of playback speed/pauses
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        else:
            # Use System Time for Webcam
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
        
        # Draw Locked Crop Box
        if 'bbox' in landmarks_dict:
            # Fix: Scale bbox to match display_frame resolution
            bbox = landmarks_dict['bbox']
            scaled_bbox = [int(coord * scale_disp) for coord in bbox]
            pose_estimator.draw_bbox(display_frame, scaled_bbox)

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
                # Use Robust Calibration (Handles Side Views & Partial Visibility)
                # Pass 'results' (pose_results) and mp_pose enum
                vbt_analyzer.attempt_robust_calibration(landmarks_dict, results, pose_estimator.mp_pose)
                
                # If still None (e.g. waiting for valid frame), execute legacy or wait
                # Actually robust calib sets a default if needed, so it shouldn't hang forever unless no landmarks found.
                if vbt_analyzer.calibration_factor is None:
                     # If webcam, we might want 'attempt_auto_calibration' (stand still logic)?
                     # But 'robust' is better for "Setup-Free".
                     # Let's fallback to current webcam logic ONLY if robust didn't fire?
                     # No, robust is better.
                     pass
            else:
                velocity = vbt_analyzer.calculate_velocity(bar_y, current_time)
                vbt_analyzer.process_rep(velocity)

        # ----------------------------------------------------
        # 3. Update Graph & UI
        # ----------------------------------------------------
        graph_history.append(velocity)
        if len(graph_history) > max_graph_points: graph_history.pop(0)

        # Draw Pause Status
        if paused:
            cv2.putText(display_frame, "PAUSED", (display_frame.shape[1] - 150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Drawing UI on 'display_frame'
        ui_y = 30
        
        # Calibration Status
        if vbt_analyzer.calibration_factor:
            cal_text = f"Calibration: OK ({args.grip_finger.capitalize()})"
            cal_color = (0, 255, 0)
        else:
            cal_text = "Calibrating..."
            cal_color = (0, 255, 255)
        draw_text(display_frame, cal_text, (10, ui_y), color=cal_color)
        ui_y += 40
        
        # Rep Count
        state_text = vbt_analyzer.current_state
        if state_text == "CONCENTRIC":
            state_text = "LIFTING"
        elif state_text == "ECCENTRIC":
            state_text = "DOWN"
            
        rep_text = f"Reps: {vbt_analyzer.rep_count}  [{state_text}]"
        draw_text(display_frame, rep_text, (10, ui_y), (0, 255, 0))
        ui_y += 40
        
        # Velocity
        if vbt_analyzer.calibration_factor:
            fatigue = vbt_analyzer.get_fatigue_status()
            col = get_color_by_fatigue(fatigue)
            draw_text(display_frame, f"Vel: {velocity:.2f} m/s", (10, ui_y), col)
            ui_y += 35
            if vbt_analyzer.rep_mean_velocities:
                 # Show Mean Velocity as primary metric for Last Rep
                 draw_text(display_frame, f"Last Mean: {vbt_analyzer.rep_mean_velocities[-1]:.2f} m/s", (10, ui_y), col)
                 ui_y += 35
                 # Show Peak Velocity as secondary
                 draw_text(display_frame, f"(Peak: {vbt_analyzer.rep_velocities[-1]:.2f} m/s)", (10, ui_y), (200, 200, 200), font_scale=0.5)

        # Graph
        d_h, d_w, _ = display_frame.shape
        draw_graph(display_frame, graph_history, 10, d_h - 160, 300, 150)

        cv2.imshow('VBT Hybrid Analyzer', display_frame)

        # Event Handling
        # If paused, wait indefinitely (0) until key press to reduce CPU usage?
        # But we want to loop to keep UI responsive? OpenCV waitKey(0) blocks everything.
        # Better to use waitKey(1) always for simple UI loop, or toggling?
        # User requirement said: "waitKey wait time... 1 if play, 0 if pause".
        # This implies blocking wait is okay/desired. 
        # CAUTION: If we block, we can't redraw unless we loop.
        # But if paused, image doesn't change, so blocking is fine.
        
        delay = 0 if paused else 1
        # Use waitKeyEx to capture arrow keys on Windows (which are extended codes)
        key = cv2.waitKeyEx(delay)
        
        # Windows Arrow Key Codes
        KEY_LEFT = 2424832
        KEY_RIGHT = 2555904
        
        if key == ord('q') or key == 27: # q or ESC
            break
        elif key == 32: # Space key
            paused = not paused
        elif key == ord('n') or key == KEY_RIGHT: # Next frame (n or Right Arrow)
            if paused:
                force_one_frame = True
        elif key == ord('p') or key == KEY_LEFT: # Previous frame (p or Left Arrow)
            if paused and args.video:
                curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_pos - 2))
                force_one_frame = True
            
        # Check if window is closed by user (X button)
        try:
             if cv2.getWindowProperty('VBT Hybrid Analyzer', cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
             pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
