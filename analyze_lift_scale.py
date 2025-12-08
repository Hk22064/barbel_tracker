import cv2
import argparse
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import find_peaks, savgol_filter
from collections import deque

class BarbellTracker:
    """カルマンフィルタを用いたバーベルト ラッカー"""
    def __init__(self):
        # 状態ベクトル: [x, y, dx, dy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.prediction = np.zeros((2, 1), np.float32)
        self.initialized = False

    def update(self, coord):
        measurement = np.array([[np.float32(coord[0])], [np.float32(coord[1])]])
        if not self.initialized:
            self.kf.statePre = np.array([[measurement[0,0]], [measurement[1,0]], [0], [0]], np.float32)
            self.kf.statePost = self.kf.statePre
            self.prediction = measurement
            self.initialized = True
        else:
            self.kf.correct(measurement)
        
        self.prediction = self.kf.predict()
        return (int(self.prediction[0,0]), int(self.prediction[1,0]))

    def predict_only(self):
        if not self.initialized:
            return None
        self.prediction = self.kf.predict()
        return (int(self.prediction[0,0]), int(self.prediction[1,0]))

def calculate_velocity_in_meters(coords, fps, scale_factor_history):
    """
    各フレーム間の速度を計算する (m/s)
    scale_factor_history: 各フレームごとの「1ピクセルが何メートルか」のリスト
    """
    velocities = []
    if len(coords) < 2:
        return velocities

    # スケール係数の代表値（中央値）を取得して全体に適用するか、
    # フレームごとの変動を許容するか。ここでは安定のため全体の平均/中央値を使う
    valid_scales = [s for s in scale_factor_history if s > 0]
    if not valid_scales:
        print("Warning: No valid scale factor found. Defaulting to 1.0 (Pixel unit).")
        avg_scale = 1.0
    else:
        avg_scale = np.median(valid_scales)
        print(f"DEBUG: Applied Scale Factor: 1 pixel = {avg_scale:.6f} meters")

    for i in range(1, len(coords)):
        p1 = coords[i-1]
        p2 = coords[i]
        if p1 is None or p2 is None:
            velocities.append(0)
            continue
            
        # ピクセル距離
        pixel_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # メートル距離に変換
        meter_dist = pixel_dist * avg_scale
        
        # 速度 (m/s) = 距離(m) * FPS
        velocity = meter_dist * fps
        velocities.append(velocity)
        
    return velocities, avg_scale

def count_reps(y_coords, prominence_threshold=10):
    if not y_coords:
        return 0, []
    inverted_y = [-y for y in y_coords if y is not None]
    if not inverted_y:
        return 0, []
    peaks, _ = find_peaks(inverted_y, prominence=prominence_threshold)
    return len(peaks), peaks

def apply_savgol_filter(data, window_length=15, polyorder=3):
    if len(data) < window_length:
        return data
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(data, window_length, polyorder)

def get_lifter_info(person_keypoints):
    shoulders = person_keypoints[[5, 6]]
    hips = person_keypoints[[11, 12]]
    wrists = person_keypoints[[9, 10]]
    
    visible_shoulders = shoulders[shoulders[:, 2] > 0]
    visible_hips = hips[hips[:, 2] > 0]
    visible_wrists = wrists[wrists[:, 2] > 0]
    
    lifter_center = None
    if len(visible_shoulders) > 0 and len(visible_hips) > 0:
        shoulder_mid = np.mean(visible_shoulders[:, :2], axis=0)
        hip_mid = np.mean(visible_hips[:, :2], axis=0)
        lifter_center = (shoulder_mid + hip_mid) / 2
        lifter_center = lifter_center.astype(int)

    wrist_coords = []
    if len(visible_wrists) > 0:
        for w in visible_wrists:
            wrist_coords.append(w[:2])

    return lifter_center, wrist_coords

def main(clump_model_path, video_path, output_path=None, conf_threshold=0.25, jump_threshold=100, plate_diameter_cm=45.0):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        filename = f"output_{base_name}_meter_scale.mp4"
        output_path = os.path.join(output_dir, filename)

    clump_model = YOLO(clump_model_path)
    pose_model = YOLO('yolo11m-pose.pt') 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trajectory_coords = []      
    scale_factor_history = []   # 各フレームのスケール係数 (m/px)
    trajectory_points = deque(maxlen=128)
    
    tracker = BarbellTracker()
    missing_frames = 0
    MAX_MISSING_FRAMES = 15

    # フィルタ設定
    track_history = {} 
    STATIC_VAR_THRESHOLD = 5.0 
    STATIC_HISTORY_LEN = 15
    EDGE_MARGIN_RATIO = 0.05   
    MAX_Y_DIFF_RATIO = 0.15
    
    # プレート直径 (メートル)
    PLATE_DIAMETER_M = plate_diameter_cm / 100.0

    print(f"Starting analysis. Plate Reference Size: {plate_diameter_cm} cm")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. 姿勢推定 ---
        pose_results = pose_model(frame, verbose=False)
        lifter_center = None
        wrist_coords = []
        all_keypoints = []

        if pose_results[0].keypoints and len(pose_results[0].keypoints.data) > 0:
            person_keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
            lifter_center, wrist_coords = get_lifter_info(person_keypoints)
            all_keypoints = person_keypoints 

        # --- 2. 物体検出 ---
        clump_results = clump_model.track(frame, persist=True, verbose=False, conf=conf_threshold)
        
        weight_clumps = []
        ignored_clumps = [] 

        for r in clump_results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: 
                    xyxy = box.xyxy[0].cpu().numpy()
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    
                    if center_x < width * EDGE_MARGIN_RATIO or center_x > width * (1 - EDGE_MARGIN_RATIO):
                        ignored_clumps.append({'box': xyxy, 'reason': 'Edge'})
                        continue

                    is_static = False
                    if box.id is not None:
                        track_id = int(box.id.item())
                        if track_id not in track_history:
                            track_history[track_id] = deque(maxlen=STATIC_HISTORY_LEN)
                        track_history[track_id].append(center_y)

                        if len(track_history[track_id]) >= STATIC_HISTORY_LEN:
                            y_variance = np.var(track_history[track_id])
                            if y_variance < STATIC_VAR_THRESHOLD:
                                is_static = True
                        else:
                            is_static = True 
                    
                    if is_static:
                        ignored_clumps.append({'box': xyxy, 'reason': 'Static'})
                        continue

                    weight_clumps.append({'box': xyxy, 'center': (center_x, center_y)})

        # --- 3. 候補選定 ---
        detected_mid_point = None
        valid_clumps = []
        current_scale = 0.0 # このフレームでのスケール係数

        if len(weight_clumps) >= 2:
            # スコア計算
            for wc in weight_clumps:
                area = (wc['box'][2] - wc['box'][0]) * (wc['box'][3] - wc['box'][1])
                wc['score'] = area

            if wrist_coords:
                for wc in weight_clumps:
                    min_dist_to_wrist = float('inf')
                    for w in wrist_coords:
                        d = np.linalg.norm(np.array(wc['center']) - w)
                        if d < min_dist_to_wrist:
                            min_dist_to_wrist = d
                    if min_dist_to_wrist < width * 0.15:
                        wc['score'] *= 3.0
            elif lifter_center is not None:
                 for wc in weight_clumps:
                    dist_to_center = np.linalg.norm(np.array(wc['center']) - lifter_center)
                    wc['score'] /= (1 + dist_to_center * 0.01)

            weight_clumps.sort(key=lambda wc: wc['score'], reverse=True)
            candidates = weight_clumps[:4]
            
            best_pair = []
            min_y_diff = float('inf')

            import itertools
            for c1, c2 in itertools.combinations(candidates, 2):
                if abs(c1['center'][0] - c2['center'][0]) < width * 0.1:
                    continue
                y_diff = abs(c1['center'][1] - c2['center'][1])
                if y_diff > height * MAX_Y_DIFF_RATIO:
                    continue

                if y_diff < min_y_diff:
                    min_y_diff = y_diff
                    pair = sorted([c1, c2], key=lambda x: x['center'][0])
                    best_pair = pair
            
            if best_pair:
                valid_clumps = best_pair
                c1 = valid_clumps[0]['center']
                c2 = valid_clumps[1]['center']
                detected_mid_point = (int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2))

                # ★★★ ピクセル/メートル変換係数の計算 ★★★
                # 2つのプレートの幅(ピクセル)を取得して平均する
                w1 = valid_clumps[0]['box'][2] - valid_clumps[0]['box'][0]
                w2 = valid_clumps[1]['box'][2] - valid_clumps[1]['box'][0]
                avg_plate_width_px = (w1 + w2) / 2.0
                
                if avg_plate_width_px > 0:
                    current_scale = PLATE_DIAMETER_M / avg_plate_width_px

        # スケール履歴に追加（検出できなかったときは0を入れるか、直前を使うかだが、ここでは計算用リストに入れる）
        # 0の場合は後で除外する
        scale_factor_history.append(current_scale)

        # --- 4. Kalman Filter Update / Predict ---
        final_point = None

        if detected_mid_point:
            final_point = tracker.update(detected_mid_point)
            missing_frames = 0
        else:
            if missing_frames < MAX_MISSING_FRAMES:
                final_point = tracker.predict_only()
                missing_frames += 1
            else:
                final_point = None

        trajectory_coords.append(final_point)
        if final_point:
            trajectory_points.append(final_point)

        # --- 描画 ---
        if len(all_keypoints) > 0:
            for x, y, conf in all_keypoints:
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1) 
        
        for w in wrist_coords:
             cv2.circle(frame, (int(w[0]), int(w[1])), 6, (255, 0, 0), -1)

        for clump in valid_clumps:
            b = clump['box']
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            # スケール情報の表示
            width_px = b[2] - b[0]
            cv2.putText(frame, f"{width_px:.0f}px", (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        for i in range(1, len(trajectory_points)):
            if trajectory_points[i-1] is not None and trajectory_points[i] is not None:
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)
        
        if final_point:
            cv2.circle(frame, final_point, 8, (0, 0, 255), -1)
        
        out.write(frame)

    # --- 結果分析 (m/s) ---
    y_coords = [p[1] for p in trajectory_coords if p is not None]
    
    if not y_coords:
        print("Could not detect valid trajectory.")
    else:
        smoothed_y = apply_savgol_filter(y_coords, window_length=15, polyorder=3)
        num_reps, peak_indices = count_reps(list(smoothed_y), prominence_threshold=15)
        print(f"Total Reps Detected: {num_reps}")

        # ★★★ 単位変換ありの速度計算 ★★★
        all_velocities, final_scale = calculate_velocity_in_meters(trajectory_coords, fps, scale_factor_history)
        
        # スムージング
        if len(all_velocities) > 15:
            all_velocities = apply_savgol_filter(all_velocities, window_length=9, polyorder=2)

        valid_velocities = [v for p, v in zip(trajectory_coords, [0] + list(all_velocities)) if p is not None]
        
        rep_start_idx = 0
        for i, peak_idx in enumerate(peak_indices):
            next_peak_idx = peak_indices[i+1] if i + 1 < len(peak_indices) else len(y_coords)
            v_slice = valid_velocities[rep_start_idx:next_peak_idx]

            if len(v_slice) > 0:
                avg_velocity = np.mean(v_slice)
                max_velocity = np.max(v_slice)
                # 単位を m/s で表示
                print(f"Rep {i+1}: Peak Velocity = {max_velocity:.2f} m/s, Avg Velocity = {avg_velocity:.2f} m/s")
            
            rep_start_idx = peak_idx + 1
            
    cap.release()
    out.release()
    print(f"Analysis complete. Output video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze lift with Pixel-to-Meter conversion.")
    parser.add_argument("clump_model_path", type=str, help="Path to the trained weight_clump model (.pt file).")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output video.")
    parser.add_argument("--conf_threshold", type=float, default=0.15, help="Confidence threshold for clump detection.")
    parser.add_argument("--jump_threshold", type=float, default=100.0, help="Maximum allowed pixel jump per frame.")
    parser.add_argument("--plate_diameter", type=float, default=45.0, help="Diameter of the weight plate in cm (Standard Olympic is 45.0).")
    
    args = parser.parse_args()
    main(args.clump_model_path, args.video_path, args.output_path, args.conf_threshold, args.jump_threshold, args.plate_diameter)