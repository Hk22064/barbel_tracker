
import cv2
import time
import argparse
import numpy as np
import sys
import os

# Add project root to sys.path
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yolo11_mediapipe_estimator import Yolo11MediaPipeEstimator
from src.vbt_analyzer import VBTAnalyzer

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

class VBTReviewer:
    def __init__(self, video_path, model_path, margin, smooth, threshold, filter_type="average"):
        self.video_path = video_path
        self.model_path = model_path
        self.margin = margin
        self.smooth = smooth
        self.threshold = threshold
        self.filter_type = filter_type
        self.grip_finger = 'middle' # Defaults, will likely be overridden if passed in init
        
    def set_grip(self, grip):
        self.grip_finger = grip
        
        self.frame_data = [] # Stores dict for each frame
        
    def analyze(self):
        print(f"Starting Pre-Analysis of {os.path.basename(self.video_path)}...")
        print("Please wait...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        estimator = Yolo11MediaPipeEstimator(model_path=self.model_path, margin_ratio=self.margin)
        analyzer = VBTAnalyzer(smoothing_window=self.smooth, velocity_threshold=self.threshold, filter_type=self.filter_type, grip_finger=self.grip_finger)
        
        frame_idx = 0
        velocity_log = [] # For graph drawing
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Timestamp (seconds)
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # --- Detection & Pose ---
            landmarks_dict, results = estimator.process_frame(frame)
            
            # --- VBT Logic ---
            velocity = 0.0
            bar_y = 0
            
            if 'left_wrist' in landmarks_dict and 'right_wrist' in landmarks_dict:
                lw = landmarks_dict['left_wrist']
                rw = landmarks_dict['right_wrist']
                bar_y = (lw[1] + rw[1]) / 2.0
                
                if analyzer.calibration_factor is None:
                    analyzer.attempt_robust_calibration(landmarks_dict, results, estimator.mp_pose)
                
                if analyzer.calibration_factor is not None:
                    velocity = analyzer.calculate_velocity(bar_y, current_time)
                    analyzer.process_rep(velocity)
            
            # --- Store Data ---
            velocity_log.append(velocity)
            if len(velocity_log) > 200: velocity_log.pop(0)
            
            fatigue = analyzer.get_fatigue_status()
            
            # State text map
            state_text = analyzer.current_state
            if state_text == "CONCENTRIC": state_text = "LIFTING"
            elif state_text == "ECCENTRIC": state_text = "DOWN"

            last_peak = 0.0
            last_mean = 0.0
            if analyzer.rep_velocities:
                last_peak = analyzer.rep_velocities[-1]
            if analyzer.rep_mean_velocities:
                last_mean = analyzer.rep_mean_velocities[-1]

            data = {
                "idx": frame_idx,
                "bbox": landmarks_dict.get('bbox'),
                "results": results, # MediaPipe results object (might be large? No, just normalized coords)
                "velocity": velocity,
                "rep_count": analyzer.rep_count,
                "state_text": state_text,
                "calibration_factor": analyzer.calibration_factor,
                "fatigue": fatigue,
                "last_peak": last_peak,
                "last_mean": last_mean,
                "graph_history": list(velocity_log) # Copy list
            }
            self.frame_data.append(data)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Analyzed {frame_idx}/{total_frames} frames...", end='\r')
        
        cap.release()
        estimator.close()
        print(f"\nAnalysis Complete. {len(self.frame_data)} frames processed.")

    def run_player(self):
        print("Starting Review Player...")
        print("Controls: SPACE=Pause/Play, RIGHT=Next, LEFT=Prev, Q=Quit")
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = len(self.frame_data)
        
        current_idx = 0
        paused = True # Start paused? Or playing? Let's start Playing.
        paused = False
        
        while True:
            # Loop video? Or stop at end? Stop at end.
            if current_idx >= total_frames:
                current_idx = total_frames - 1
                paused = True
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize for display
            display_frame = frame.copy()
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
                display_frame = cv2.resize(frame, (new_w, new_h))
            
            # Get Data
            data = self.frame_data[current_idx]
            
            # --- Draw Overlays ---
            # 1. BBox
            if data['bbox']:
                scaled_bbox = [int(c * scale_disp) for c in data['bbox']]
                cv2.rectangle(display_frame, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), (255, 0, 0), 2)
            
            # 2. Pose (Need to use MP drawing utils, but we don't have the estimator instance here)
            # We can re-instantiate or just use generic drawing if available?
            # Or pass the estimator to the player?
            # Actually, we stored 'results'. We can use mp.solutions.drawing_utils if we import it.
            # But simpler: Just simple lines or circles? 
            # Re-using Yolo11MediaPipeEstimator.draw_landmarks is best but it's an instance method.
            # Let's just create a dummy instance or static method?
            # For now, let's skip complex pose drawing or just basic distinct points?
            # Users want to see the stick figure. 
            # Let's instantiate a lightweight helper or just use the raw coordinates if possible.
            # Actually, `Yolo11MediaPipeEstimator` imports mp.solutions.drawing_utils. 
            # We can use that directly.
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            mp_drawing_styles = mp.solutions.drawing_styles
            
            if data['results'] and data['results'].pose_landmarks:
                 mp_drawing.draw_landmarks(
                    display_frame,
                    data['results'].pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # 3. UI Text
            ui_y = 30
            
            # Calibration
            if data['calibration_factor']:
                cal_text = "Calibration: OK"
                cal_color = (0, 255, 0)
            else:
                cal_text = "Calibrating..."
                cal_color = (0, 255, 255)
            draw_text(display_frame, cal_text, (10, ui_y), cal_color)
            ui_y += 40
            
            # Reps & State
            draw_text(display_frame, f"Reps: {data['rep_count']} [{data['state_text']}]", (10, ui_y), (0, 255, 0))
            ui_y += 40
            
            # Velocity
            col = get_color_by_fatigue(data['fatigue'])
            draw_text(display_frame, f"Vel: {data['velocity']:.2f} m/s", (10, ui_y), col)
            ui_y += 35
            
            if data['last_mean'] > 0:
                 draw_text(display_frame, f"Last Mean: {data['last_mean']:.2f} m/s", (10, ui_y), col)
                 ui_y += 35
                 draw_text(display_frame, f"(Peak: {data['last_peak']:.2f} m/s)", (10, ui_y), (200, 200, 200), font_scale=0.5)

            # Paused Indicator
            if paused:
                cv2.putText(display_frame, "PAUSED", (display_frame.shape[1] - 150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Frame Info
            cv2.putText(display_frame, f"Frame: {current_idx}/{total_frames}", (display_frame.shape[1] - 200, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Graph
            d_h, d_w, _ = display_frame.shape
            draw_graph(display_frame, data['graph_history'], 10, d_h - 160, 300, 150)

            cv2.imshow('VBT Review Player', display_frame)
            
            # Input Handling
            delay = 0 if paused else 1 # Wait indefinitely if paused? No, we need loop to catch resizing or window events? 
            # Actually 0 blocks forever until key. 1 waits 1ms.
            # If paused, we want to block until key press to prevent high CPU, 
            # BUT we also want to handle window close.
            # waitKeyEx(0) is fine for "Paused".
            
            wait_ms = 0 if paused else 20 # ~30fps -> 33ms. Processing takes time, so 20ms is safe.
            if paused: wait_ms = 0
            
            key = cv2.waitKeyEx(wait_ms)
            
            # Arrow Keys
            KEY_LEFT = 2424832
            KEY_RIGHT = 2555904
            
            if key == ord('q') or key == 27:
                break
            elif key == 32: # Space
                paused = not paused
            elif key == ord('n') or key == KEY_RIGHT: # Next
                current_idx += 1
                if current_idx >= total_frames: current_idx = total_frames - 1
                if paused: 
                    pass # Just update index and loop will redraw
            elif key == ord('p') or key == KEY_LEFT: # Prev
                current_idx -= 1
                if current_idx < 0: current_idx = 0
                if paused:
                    pass
            
            # If playing, advance
            if not paused:
                current_idx += 1
                
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="VBT Video Review Tool")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path")
    
    # Tuning Parameters
    parser.add_argument("--margin", type=float, default=0.3, help="ROI Margin Ratio (default: 0.3)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing Window (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Velocity Threshold (default: 0.05)")
    parser.add_argument("--filter_type", type=str, default="average", help="Filter Type (average, kalman, butterworth)")
    parser.add_argument("--grip_finger", type=str, default="middle", help="Grip finger on 81cm line")
    args = parser.parse_args()

    reviewer = VBTReviewer(
        video_path=args.video, 
        model_path=args.model,
        margin=args.margin,
        smooth=args.smooth,
        threshold=args.threshold,
        filter_type=args.filter_type
    )
    reviewer.set_grip(args.grip_finger)
    reviewer.analyze()
    reviewer.run_player()

if __name__ == "__main__":
    main()
