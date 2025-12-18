import cv2
import time
import numpy as np
import os
from VBT_mediapipe.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
from VBT_mediapipe.vbt_analyzer import VBTAnalyzer

# Script Configuration
INPUT_VIDEO = "VBT_mediapipe/video/vertical/front_10rep.mp4"
OUTPUT_VIDEO = "experiment_results/demo_hybrid_front_10rep.mp4"

# --- Visualization Helpers (Copied from app_yolo11.py) ---
def draw_text(img, text, pos, color=(0, 255, 0), font_scale=0.7, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_graph(image, velocity_history, graph_x, graph_y, graph_w, graph_h, max_val=2.0):
    # Draw background
    # Use alpha blending for background if possible? For now solid is fine.
    overlay = image.copy()
    cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image) # Semi-transparent background
    
    if len(velocity_history) < 2:
        return

    points = []
    for i, val in enumerate(velocity_history):
        val = max(0, min(val, max_val))
        x = graph_x + int((i / len(velocity_history)) * graph_w)
        y = graph_y + graph_h - int((val / max_val) * graph_h)
        points.append((x, y))

    if points:
        cv2.polylines(image, [np.array(points)], isClosed=False, color=(0, 255, 255), thickness=2)

def get_color_by_fatigue(drop_off_percent):
    if drop_off_percent < 10:
        return (0, 255, 0) # Green
    elif drop_off_percent < 20:
        return (0, 255, 255) # Yellow
    else:
        return (0, 0, 255) # Red

def main():
    print(f"Generating Demo Video from: {INPUT_VIDEO}")
    
    if not os.path.exists(INPUT_VIDEO):
        print("Error: Input video not found.")
        return

    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    # Check dimensions
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine if rotation is needed (Simple heuristic: If it looks landscape but should be vertical)
    # front_10rep is known to be vertical.
    ROTATE = False
    if cap_width > cap_height:
        print("Detected Landscape orientation for Vertical video. Applying 90-degree Rotation.")
        ROTATE = True
        width = cap_height
        height = cap_width
    else:
        width = cap_width
        height = cap_height
    
    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Initialize Models
    # Recalling that front_10rep needs lower confidence for "Far" view
    print("Initializing Hybrid Estimator...")
    pose_estimator = Yolo11MediaPipeEstimator(model_path="yolo11n.pt", min_detection_confidence=0.3, min_tracking_confidence=0.3)
    vbt_analyzer = VBTAnalyzer(smoothing_window=5)
    
    frame_count = 0
    graph_history = []
    max_graph_points = 200

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply Rotation if needed
        if ROTATE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        frame_count += 1
        current_time_in_video = frame_count / fps # Simulation time
        
        # 1. Process
        landmarks_dict, results = pose_estimator.process_frame(frame)
        
        # 2. Draw Skeleton & Box
        pose_estimator.draw_landmarks(frame, results)
        if 'bbox' in landmarks_dict and landmarks_dict['bbox']:
             pose_estimator.draw_bbox(frame, landmarks_dict['bbox'])

        # 3. Analyze
        velocity = 0.0
        if 'left_wrist' in landmarks_dict and 'right_wrist' in landmarks_dict:
            lw = landmarks_dict['left_wrist']
            rw = landmarks_dict['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            # Use Video Mode Calibration Strategy (Immediate)
            if vbt_analyzer.calibration_factor is None:
                vbt_analyzer.calibrate(lw, rw)
            else:
                velocity = vbt_analyzer.calculate_velocity(bar_y, current_time_in_video)
                vbt_analyzer.process_rep(velocity)
        
        # 4. Update Graph
        graph_history.append(velocity)
        if len(graph_history) > max_graph_points:
            graph_history.pop(0)
            
        # 5. Draw UI
        ui_y = 50
        
        # Status
        cal_status = "Calibration: OK" if vbt_analyzer.calibration_factor else "Calibrating..."
        draw_text(frame, cal_status, (20, ui_y), color=(0, 255, 0) if vbt_analyzer.calibration_factor else (0, 255, 255))
        ui_y += 40
        
        # Reps
        rep_text = f"Reps: {vbt_analyzer.rep_count}  [{vbt_analyzer.current_state}]"
        # Color based on state
        state_color = (200, 200, 200)
        if vbt_analyzer.current_state == vbt_analyzer.STATE_CONCENTRIC:
            state_color = (0, 255, 0)
        elif vbt_analyzer.current_state == vbt_analyzer.STATE_ECCENTRIC:
            state_color = (0, 255, 255)
            
        draw_text(frame, rep_text, (20, ui_y), color=state_color)
        ui_y += 40
        
        # Velocity & Fatigue
        if vbt_analyzer.calibration_factor:
            fatigue = vbt_analyzer.get_fatigue_status()
            color = get_color_by_fatigue(fatigue)
            
            draw_text(frame, f"Vel: {velocity:.2f} m/s", (20, ui_y), color=color, font_scale=1.2, thickness=3)
            ui_y += 50
            
            if vbt_analyzer.rep_velocities:
                 draw_text(frame, f"Last Peak: {vbt_analyzer.rep_velocities[-1]:.2f} m/s", (20, ui_y), color=color)
                 ui_y += 40
                 draw_text(frame, f"Drop-off: {fatigue:.1f}%", (20, ui_y), color=color)

        # Graph (Bottom Right)
        draw_graph(frame, graph_history, 20, height - 200, 400, 150)
        
        # Write Frame
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    pose_estimator.close()
    
    elapsed = time.time() - start_time
    print(f"Done! Saved to {OUTPUT_VIDEO}")
    print(f"Total time: {elapsed:.2f}s, Processed FPS: {frame_count/elapsed:.1f}")

if __name__ == "__main__":
    main()
