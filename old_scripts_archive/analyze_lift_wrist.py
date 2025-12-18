import cv2
import numpy as np
import os
import sys
import traceback
from ultralytics import YOLO
from scipy.signal import savgol_filter
from enum import Enum, auto

# --- State Machine & Utils ---

class BarbellState(Enum):
    IDLE = auto()
    DESCENDING = auto()
    BOTTOM = auto()
    ASCENDING = auto()
    TOP = auto()

def apply_savgol_filter(data, window_length=15, polyorder=3):
    # データ数が少なすぎる場合はフィルタをかけずにそのまま返す
    if len(data) < window_length:
        return data
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(data, window_length, polyorder)

def count_reps_state_machine(y_coords):
    state = BarbellState.IDLE
    rep_count = 0
    
    # 【修正箇所】NumPy配列に対して "if y_coords:" はエラーになるため、len()で判定する
    if len(y_coords) > 0:
        start_y = y_coords[0]
    else:
        start_y = 0
        
    bottom_y = 0
    
    DESCEND_THRESH = 20
    ASCEND_THRESH = 20
    
    for i in range(1, len(y_coords)):
        curr_y = y_coords[i]
        prev_y = y_coords[i-1]
        
        if state == BarbellState.IDLE:
            if curr_y - start_y > DESCEND_THRESH: 
                state = BarbellState.DESCENDING

        elif state == BarbellState.DESCENDING:
            if curr_y < prev_y: 
                state = BarbellState.BOTTOM
                bottom_y = prev_y
            elif curr_y > bottom_y: 
                bottom_y = curr_y

        elif state == BarbellState.BOTTOM:
            if bottom_y - curr_y > ASCEND_THRESH:
                state = BarbellState.ASCENDING
            elif curr_y > bottom_y:
                state = BarbellState.DESCENDING 
                bottom_y = curr_y

        elif state == BarbellState.ASCENDING:
            # 元の位置に戻ったか判定
            if abs(curr_y - start_y) < 30 or curr_y < start_y: 
                rep_count += 1
                state = BarbellState.TOP
                start_y = curr_y

        elif state == BarbellState.TOP:
            state = BarbellState.IDLE
            start_y = curr_y

    return rep_count

# --- Main Analysis Logic ---

def calculate_velocity(y_coords, fps, scale):
    velocities = []
    if len(y_coords) == 0:
        return []
    
    for i in range(1, len(y_coords)):
        curr_y = float(y_coords[i])
        prev_y = float(y_coords[i-1])
        dy = prev_y - curr_y 
        v = dy * scale * fps
        velocities.append(v)
    return velocities 

def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("video_path", type=str)
    parser.add_argument("--scale", type=float, default=0.0016, help="Meters per pixel")
    parser.add_argument("--export_npy", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path
    video_path = args.video_path
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    model_suffix = os.path.basename(model_path).replace('.pt','')
    output_path = os.path.join(output_dir, f"output_{base_name}_wrist_{model_suffix}.mp4")

    print(f"--- Wrist Tracking Analysis ---")
    start_time = time.time()
    
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        wrist_y_history = []
        wrist_distances = [] # Store distances between wrists
        points_to_draw = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            
            # Safe Keypoint Extraction
            kpts = None
            if results and results[0].keypoints is not None and results[0].keypoints.data is not None:
                if len(results[0].keypoints.data) > 0:
                     kpts = results[0].keypoints.data[0].cpu().numpy()

            midpoint = None
            if kpts is not None and len(kpts) > 10:
                l_wrist = kpts[9]
                r_wrist = kpts[10]
                if l_wrist[2].item() > 0.3 and r_wrist[2].item() > 0.3:
                    mid_y = (l_wrist[1].item() + r_wrist[1].item()) / 2
                    midpoint = (int((l_wrist[0].item()+r_wrist[0].item())/2), int(mid_y))
                    
                    # Calculate wrist distance
                    w_dist = np.linalg.norm(l_wrist[:2] - r_wrist[:2])
                    wrist_distances.append(w_dist)

                    wrist_y_history.append(mid_y)
                    points_to_draw.append(midpoint)
                    
                    cv2.circle(frame, (int(l_wrist[0].item()), int(l_wrist[1].item())), 5, (255,0,0), -1)
                    cv2.circle(frame, midpoint, 8, (0,255,255), -1)
                else:
                    if wrist_y_history: wrist_y_history.append(wrist_y_history[-1])
                    else: wrist_y_history.append(height/2)
            else:
                 if wrist_y_history: wrist_y_history.append(wrist_y_history[-1])
                 else: wrist_y_history.append(height/2)
            
            if len(points_to_draw) > 1:
                for i in range(1, len(points_to_draw)):
                     cv2.line(frame, points_to_draw[i-1], points_to_draw[i], (0, 255, 255), 2)

            out.write(frame)

        cap.release()
        out.release()
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Total Time: {elapsed:.2f} s")
        print(f"Average FPS: {total_frames / elapsed:.2f}")

        # Post Processing
        if not wrist_y_history:
            print("No wrist data collected.")
            return

        smoothed_y = apply_savgol_filter(wrist_y_history)
        reps = count_reps_state_machine(smoothed_y)
        
        # Auto-Calibration (Wrist Distance = 81cm)
        final_scale = args.scale # Default / Fallback
        
        if wrist_distances:
            median_dist_px = np.median(wrist_distances)
            if median_dist_px > 0:
                auto_scale = 0.81 / median_dist_px
                print(f"[Auto-Calibration] Median Wrist Dist: {median_dist_px:.2f} px")
                print(f"[Auto-Calibration] Calculated Scale: {auto_scale:.6f} m/px (Ref: 0.81m)")
                final_scale = auto_scale
            else:
                print(f"[Warning] Median wrist distance is 0. Using default scale: {final_scale}")
        else:
             print(f"[Warning] No wrist distance data. Using default scale: {final_scale}")

        # Velocity
        velocities = calculate_velocity(smoothed_y, fps, final_scale)
        
        if args.export_npy:
            np.save(args.export_npy, velocities)
            print(f"Exported velocity to {args.export_npy}")
        
        print(f"Total Reps Detected (Wrist): {reps}")
        print(f"Output saved to: {output_path}")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()