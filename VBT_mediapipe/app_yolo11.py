"""
==============================================================================
VBT (Velocity Based Training) Analyzer - YOLO11 Hybrid Mode
==============================================================================

【実行手順 / Execution Instructions】

1. Conda環境の有効化 (Activate Conda Environment):
   conda activate yolo

2. 必要なライブラリのインストール (Install Dependencies):
   pip install mediapipe opencv-python numpy matplotlib ultralytics

3. スクリプトの実行 (Run Script):
   - Webカメラを使用する場合 (Webcam):
     python app_yolo11.py

   - 動画ファイルを使用する場合 (Video File):
     python app_yolo11.py --video data/squat.mp4

【操作方法 / Controls】
   - 'q': 終了 (Quit)
   - 'r': リセット (Reset Detection) - 人物の検出をやり直します。

==============================================================================
"""

import cv2
import time
import argparse
import numpy as np
from yolo11_hybrid_estimator import Yolo11HybridEstimator
from vbt_analyzer import VBTAnalyzer

def draw_text(img, text, pos, color=(0, 255, 0), font_scale=0.7, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_graph(image, velocity_history, graph_x, graph_y, graph_w, graph_h, max_val=2.0):
    """
    Draw a simple line graph of velocity history on the image.
    """
    # Draw background
    cv2.rectangle(image, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), -1)
    
    if len(velocity_history) < 2:
        return

    # Normalize and plot
    # max_val is the top limit of the graph in m/s
    points = []
    for i, val in enumerate(velocity_history):
        # Clamp value
        val = max(0, min(val, max_val))
        
        # X coordinate
        x = graph_x + int((i / len(velocity_history)) * graph_w)
        
        # Y coordinate (Inverted because Y is down)
        # val=0 -> y = graph_y + graph_h
        # val=max -> y = graph_y
        y = graph_y + graph_h - int((val / max_val) * graph_h)
        points.append((x, y))

    # Draw lines
    if points:
        cv2.polylines(image, [np.array(points)], isClosed=False, color=(0, 255, 255), thickness=1)

def get_color_by_fatigue(drop_off_percent):
    """
    Return color (B, G, R) based on fatigue drop-off.
    0-10% -> Green
    10-20% -> Yellow
    >20% -> Red
    """
    if drop_off_percent < 10:
        return (0, 255, 0) # Green
    elif drop_off_percent < 20:
        return (0, 255, 255) # Yellow
    else:
        return (0, 0, 255) # Red

def main():
    parser = argparse.ArgumentParser(description="VBT Analyzer (YOLO11 Hybrid)")
    parser.add_argument("--video", type=str, help="Path to video file", default=None)
    args = parser.parse_args()

    # Initialize Modules with YOLO11 Hybrid
    print("Initializing YOLO11 Hybrid Estimator...")
    pose_estimator = Yolo11HybridEstimator(model_path="yolo11n.pt") 
    
    vbt_analyzer = VBTAnalyzer(smoothing_window=5)

    # Input Source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Opening video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Opening Webcam")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Graph history buffer
    graph_history = []
    max_graph_points = 200

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Frame Resize for consistency (optional, but good for performance)
        # frame = cv2.resize(frame, (640, 480))
        
        current_time = time.time()
        
        # 1. Pose Estimation (YOLO11 -> Crop -> MediaPipe -> Global Coords)
        landmarks_dict, results = pose_estimator.process_frame(frame)
        
        # Draw Skeleton & BBox
        pose_estimator.draw_landmarks(frame, results)
        if 'bbox' in landmarks_dict:
            pose_estimator.draw_bbox(frame, landmarks_dict['bbox'])

        # 2. VBT Logic
        bar_y = 0
        velocity = 0.0
        
        if 'left_wrist' in landmarks_dict and 'right_wrist' in landmarks_dict:
            lw = landmarks_dict['left_wrist']
            rw = landmarks_dict['right_wrist']
            
            # Using average Y of wrists as bar height
            bar_y = (lw[1] + rw[1]) / 2.0
            
            # Auto Calibration
            if vbt_analyzer.calibration_factor is None:
                # Video Mode: Calibrate Instantly on first valid frame
                if args.video:
                    vbt_analyzer.calibrate(lw, rw)
                    # Reset velocity because this frame is just for calibration
                    velocity = 0.0
                else:
                    # Webcam Mode: Wait for stability
                    raw_vel = 0.0
                    if vbt_analyzer.prev_y is not None:
                         raw_vel = abs(vbt_analyzer.prev_y - bar_y) # pixels per frame roughly
                    
                    vbt_analyzer.attempt_auto_calibration(lw, rw, raw_vel)
                    
                    # Update velocity state (even if not calibrated, to update prev values)
                    vbt_analyzer.calculate_velocity(bar_y, current_time)
                    velocity = 0.0
            
            else:
                # Normal Operation
                velocity = vbt_analyzer.calculate_velocity(bar_y, current_time)
                # Analyze Rep (Automatic State Machine)
                current_rep_max = vbt_analyzer.process_rep(velocity)
            
            # Key Handling (Restored to original position)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if hasattr(pose_estimator, 'reset'):
                    pose_estimator.reset()
        else:
             if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        # 3. Update Graph Data
        graph_history.append(velocity)
        if len(graph_history) > max_graph_points:
            graph_history.pop(0)

        # 4. Draw UI
        # ---------------------------------------------------------------------
        # Status Block (Top Left)
        # ---------------------------------------------------------------------
        ui_y = 30
        
        # Calibration Status
        if vbt_analyzer.calibration_factor:
            cal_color = (0, 255, 0)
            cal_text = "Calibration: OK (Hybrid)"
            if args.video:
                 cal_text = "Calibration: OK (Hybrid/Video)"
        else:
            cal_color = (0, 255, 255)
            if args.video:
                 cal_text = "Calibrating..."
            else:
                 # visual feedback for buffer filling
                 fill_ratio = len(vbt_analyzer.calibration_buffer) / vbt_analyzer.calibration_buffer.maxlen
                 cal_text = f"Calibrating... {int(fill_ratio*100)}% (Stay Still)"
            
        draw_text(frame, cal_text, (10, ui_y), color=cal_color)
        ui_y += 40
        
        # Rep Count & Phase
        rep_text = f"Reps: {vbt_analyzer.rep_count}  [{vbt_analyzer.current_state}]"
        # Color code phase
        if vbt_analyzer.current_state == vbt_analyzer.STATE_CONCENTRIC:
            phase_color = (0, 255, 0) # Green for lifting
        elif vbt_analyzer.current_state == vbt_analyzer.STATE_ECCENTRIC:
            phase_color = (0, 255, 255) # Yellow for lowering
        else:
            phase_color = (200, 200, 200) # Gray for waiting
            
        draw_text(frame, rep_text, (10, ui_y), color=phase_color)
        ui_y += 40
        
        # Matrix: Velocity Display
        if vbt_analyzer.calibration_factor:
            # Determine Color based on fatigue (Compare current rep peak to first rep peak)
            fatigue_drop = vbt_analyzer.get_fatigue_status()
            vel_color = get_color_by_fatigue(fatigue_drop)
            
            # Show Current Velocity (Realtime)
            vel_text = f"Cur Vel: {velocity:.2f} m/s"
            draw_text(frame, vel_text, (10, ui_y), color=vel_color, font_scale=0.7)
            ui_y += 30

            # Show Max Velocity of Current/Last Rep
            if vbt_analyzer.rep_velocities:
                last_max = vbt_analyzer.rep_velocities[-1]
                max_text = f"Last Peak: {last_max:.2f} m/s"
                draw_text(frame, max_text, (10, ui_y), color=vel_color, font_scale=1.0)
                ui_y += 35
                
                # Fatigue
                state_text = f"Drop-off: {fatigue_drop:.1f}%"
                draw_text(frame, state_text, (10, ui_y), color=vel_color)
            else:
                 # Show ongoing max if in progress
                 pass

        # Draw Graph (Bottom Right)
        h, w, _ = frame.shape
        draw_graph(frame, graph_history, 10, h - 160, 300, 150)

        # 5. Draw UI & Display
        # Resize for display if too large (e.g. 4K or large vertical video)
        display_frame = frame
        max_h = 800
        max_w = 1280
        h, w = frame.shape[:2]
        
        scale = 1.0
        if h > max_h or w > max_w:
            scale_h = max_h / h
            scale_w = max_w / w
            scale = min(scale_h, scale_w)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
            
            
        cv2.imshow('VBT Analyzer (YOLO11 Hybrid)', display_frame)
        
        # Check if window is closed by user (X button)
        if cv2.getWindowProperty('VBT Analyzer (YOLO11 Hybrid)', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save Results on Exit
    print("Saving measurement results...")
    report_lines = vbt_analyzer.get_results_summary()
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vbt_report_hybrid_{timestamp}.txt"
    
    # Add Header Date
    report_lines.insert(1, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"Report saved to: {filename}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Cleanup
    if hasattr(pose_estimator, 'close'):
        pose_estimator.close()
    
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
