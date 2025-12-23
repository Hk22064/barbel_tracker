
import cv2
import sys
import os
import numpy as np

# Add project root and VBT_mediapipe to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VBT_mediapipe'))

from VBT_mediapipe.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
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
    video_path = os.path.join('VBT_mediapipe', 'video', 'horizontal', 'front_9rep.mp4')
    print(f"Target video path: {video_path}")
    if not os.path.exists(video_path):
        print(f"Video DOES NOT EXIST at {video_path}")
        # Callback to find it if moved
        video_path = 'test_tate.mp4' # Fallback
        if not os.path.exists(video_path):
            print("Video not found.")
            return

    # Use default params from experiment
    pose_estimator = Yolo11MediaPipeEstimator(model_path='yolo11n.pt', margin_ratio=0.3)
    vbt_analyzer = VBTAnalyzer(smoothing_window=5, velocity_threshold=0.04, filter_type='average')

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}")

    # Fast forward to middle of a rep (e.g. frame 150)
    start_frame = 150
    if start_frame >= total_frames:
        start_frame = total_frames // 2
        print(f"Start frame adjust to {start_frame}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    graph_history = []
    max_graph_points = 200
    
    # Run for 30 frames to populate graph and status
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: break
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # 1. Estimate
        display_frame = frame.copy()
        landmarks_dict, results = pose_estimator.process_frame(frame)
        
        # Draw Overlays
        pose_estimator.draw_landmarks(display_frame, results)
        if 'bbox' in landmarks_dict:
             pose_estimator.draw_bbox(display_frame, landmarks_dict['bbox'])

        # 2. Analyze
        bar_y = 0
        velocity = 0.0
        if 'left_wrist' in landmarks_dict and 'right_wrist' in landmarks_dict:
            lw = landmarks_dict['left_wrist']
            rw = landmarks_dict['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            # Use Robust Calibration
            vbt_analyzer.attempt_robust_calibration(landmarks_dict, results, pose_estimator.mp_pose)
            
            if vbt_analyzer.calibration_factor:
                 velocity = vbt_analyzer.calculate_velocity(bar_y, current_time)
                 vbt_analyzer.process_rep(velocity)
        
        # 3. Update Graph History
        graph_history.append(velocity)
        if len(graph_history) > max_graph_points: graph_history.pop(0)

        # 4. Draw UI (Simulated)
        ui_y = 30
        
        # Calibration
        if vbt_analyzer.calibration_factor:
            cal_text = "Calibration: OK (Hybrid)"
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
                 draw_text(display_frame, f"Last Mean: {vbt_analyzer.rep_mean_velocities[-1]:.2f} m/s", (10, ui_y), col)
                 ui_y += 35
                 draw_text(display_frame, f"(Peak: {vbt_analyzer.rep_velocities[-1]:.2f} m/s)", (10, ui_y), (200, 200, 200), font_scale=0.5)

        # Graph
        d_h, d_w, _ = display_frame.shape
        draw_graph(display_frame, graph_history, 10, d_h - 160, 300, 150)
    
    # Save the final processed frame
    out_dir = 'Thesis_Materials/figures'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'system_gui_view.png')
    success = cv2.imwrite(out_path, display_frame)
    print(f"Saved system view to {out_path} : Success={success}")

if __name__ == "__main__":
    main()
