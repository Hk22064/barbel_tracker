import cv2
import argparse
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import savgol_filter
from collections import deque
from enum import Enum, auto

# --- 1. State Machine for Rep Counting ---
class BarbellState(Enum):
    IDLE = auto()
    DESCENDING = auto()
    BOTTOM = auto()
    ASCENDING = auto()
    TOP = auto()

class BarbellRepCounter:
    def __init__(self, descend_threshold=0.15, ascend_threshold=0.15):
        """
        descend_threshold: 下降検知に必要な移動距離 (m)
        ascend_threshold: 上昇検知に必要な移動距離 (m)
        """
        self.state = BarbellState.IDLE
        self.rep_count = 0
        self.current_rep_peak_velocity = 0.0
        self.current_rep_velocities = []
        
        self.start_y = None      # Rep開始時の高さ (Top)
        self.bottom_y = None     # ボトムの高さ
        self.last_y = None       # 直前フレームの高さ
        
        # 閾値 (メートル単位で管理したいため、ピクセル変換後に判定するが、
        # ここでは簡易的に相対座標やピクセルで判定するか、後でメートルで判定するか。
        # リアルタイム判定は難しいので、Y座標(ピクセル)の推移で判定する形にする)
        self.descend_threshold_px = 50 # 仮の初期値 (後でスケールに合わせて調整可能)
        self.ascend_threshold_px = 50

    def process_frame(self, current_y, current_velocity_m_s, fps):
        """
        各フレームで呼び出し、状態を更新する。
        current_y: 現在のバーベルのY座標 (ピクセル, ※Yは下に行くほど大きい)
        current_velocity: 現在の速度 (m/s, 正なら移動中)
        """
        if self.last_y is None:
            self.last_y = current_y
            self.start_y = current_y
            return

        # Y座標の増分 (下方向が正)
        dy = current_y - self.last_y 
        
        # 状態遷移ロジック
        if self.state == BarbellState.IDLE:
            # ある程度 (閾値以上) 下がったら Descending 開始
            if current_y - self.start_y > 20: # 20px程度の遊び
                self.state = BarbellState.DESCENDING
                self.start_y = self.last_y # ここをRep開始点とみなす
                self.current_rep_velocities = []

        elif self.state == BarbellState.DESCENDING:
            # 下降中。速度が落ちて、かつ十分下がっていれば Bottom 候補
            # 単純に「上がり始めたら」判定でも良い
            if dy < 0: # 上がり始めた (Yが減少)
                self.state = BarbellState.BOTTOM
                self.bottom_y = self.last_y

        elif self.state == BarbellState.BOTTOM:
            # ボトムから明確に上がり始めたら Ascending
            if self.bottom_y - current_y > 10: # 10px以上上昇
                self.state = BarbellState.ASCENDING
            # もしまた下がりだしたら Descending に戻る (バウンド等)
            elif current_y > self.bottom_y:
                self.state = BarbellState.DESCENDING
                self.bottom_y = current_y

        elif self.state == BarbellState.ASCENDING:
            self.current_rep_velocities.append(current_velocity_m_s)
            
            # Rep開始位置付近まで戻ったか？
            if current_y <= self.start_y + 20: # ほぼ元の高さ
                self.state = BarbellState.TOP
        
        elif self.state == BarbellState.TOP:
            # 1 Rep 完了
            self.rep_count += 1
            self.state = BarbellState.IDLE
            # 次のRepのためにリセット
            self.start_y = current_y
            
        self.last_y = current_y
        return self.state


# --- 2. Existing Classes ---

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

def apply_savgol_filter(data, window_length=9, polyorder=3): # [Tuned] window=9 for better sensitivity
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

def count_reps_state_machine(y_coords, velocities, fps, plate_diameter_m):
    """
    全フレーム処理後にステートマシンで回数をカウントしなおす関数
    (リアルタイムではなく、スムージング後のデータを使えるため精度が高い)
    """
    state = BarbellState.IDLE
    rep_count = 0
    reps_data = [] # {'start_idx': int, 'end_idx': int, 'max_velocity': float}
    
    start_y = y_coords[0]
    bottom_y = 0
    
    # 閾値を動的に決めても良いが、ここでは固定値である程度判定
    # ピクセル単位の閾値 (後でメートル換算も可能だが入力はピクセル)
    DESCEND_THRESH = 20 # [Tuned] 30 -> 20 (感度向上)
    ASCEND_THRESH = 20 # [Tuned] 30 -> 20 (感度向上)
    
    current_rep_start_idx = 0
    current_rep_bottom_idx = 0

    for i in range(1, len(y_coords)):
        curr_y = y_coords[i]
        prev_y = y_coords[i-1]
        
        if curr_y is None or prev_y is None:
            continue

        if state == BarbellState.IDLE:
            # 基準点より下がった
            if curr_y - start_y > DESCEND_THRESH: 
                state = BarbellState.DESCENDING
                current_rep_start_idx = i

        elif state == BarbellState.DESCENDING:
            # 下降中に「上がり始めた」らボトムとみなす
            if curr_y < prev_y: 
                state = BarbellState.BOTTOM
                bottom_y = prev_y
                current_rep_bottom_idx = i-1
            elif curr_y > bottom_y: # まだ下がってるなら更新
                bottom_y = curr_y
                current_rep_bottom_idx = i

        elif state == BarbellState.BOTTOM:
            # 明確に上昇開始
            if bottom_y - curr_y > ASCEND_THRESH:
                state = BarbellState.ASCENDING
            elif curr_y > bottom_y: # リバウンド等でさらに下がった場合
                state = BarbellState.DESCENDING 
                bottom_y = curr_y

        elif state == BarbellState.ASCENDING:
            # 開始点付近に戻った
            # [Tuned] 戻り判定も少し緩くする (20px -> 30px)
            if abs(curr_y - start_y) < 30 or curr_y < start_y:
                # Rep Complete
                rep_count += 1
                state = BarbellState.TOP
                
                # このRep区間 (Bottom -> Top) の最大速度を取得
                # 上昇局面: current_rep_bottom_idx 〜 i
                v_slice = velocities[current_rep_bottom_idx:i]
                max_v = np.max(v_slice) if len(v_slice) > 0 else 0
                avg_v = np.mean(v_slice) if len(v_slice) > 0 else 0
                
                reps_data.append({
                    'rep_no': rep_count,
                    'max_velocity': max_v,
                    'avg_velocity': avg_v
                })
                
                # 次の基準点を更新 (今の位置)
                start_y = curr_y

        elif state == BarbellState.TOP:
            state = BarbellState.IDLE
            start_y = curr_y # リセット

    return rep_count, reps_data


def analyze_lift(clump_model_path, video_path, output_path=None, conf_threshold=0.25, jump_threshold=100, plate_diameter_cm=45.0, export_npy=None):
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
    scale_factor_history = []   # 全フレームのスケール係数収集用
    trajectory_points = deque(maxlen=128)
    
    tracker = BarbellTracker()
    missing_frames = 0
    MAX_MISSING_FRAMES = 15

    # フィルタ設定
    track_history = {} 
    active_ids = set() # [New Logic] Movement Trigger: IDs that have moved enough to be valid
    MOVEMENT_TRIGGER_THRESHOLD = 5.0 # [New Logic] Variance threshold to activation
    
    STATIC_VAR_THRESHOLD = 5.0 
    STATIC_HISTORY_LEN = 15
    EDGE_MARGIN_RATIO = 0.05   
    MAX_Y_DIFF_RATIO = 0.3    
    
    PLATE_DIAMETER_M = plate_diameter_cm / 100.0

    print(f"Starting analysis. Plate Reference Size: {plate_diameter_cm} cm")

    # --- Pass 1: Video Analysis & Data Collection ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pose Estimation
        pose_results = pose_model(frame, verbose=False)
        lifter_center = None
        wrist_coords = []
        all_keypoints = []

        if pose_results[0].keypoints and len(pose_results[0].keypoints.data) > 0:
            person_keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
            lifter_center, wrist_coords = get_lifter_info(person_keypoints)
            all_keypoints = person_keypoints 

        # 2. Object Detection
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
                    
                    # [New Logic] Movement Trigger
                    # Only consider objects that have shown movement (variance > threshold)
                    is_active = False
                    if box.id is not None:
                        track_id = int(box.id.item())
                        
                        # Already active?
                        if track_id in active_ids:
                            is_active = True
                        else:
                            # Update history to check for movement
                            if track_id not in track_history:
                                track_history[track_id] = deque(maxlen=STATIC_HISTORY_LEN)
                            track_history[track_id].append(center_y)
                            
                            # Check variance if history is full enough
                            if len(track_history[track_id]) >= 3: # [Tuned] 5 -> 3 (Respond faster)
                                y_variance = np.var(track_history[track_id])
                                if y_variance > 2.0: # [Tuned] 5.0 -> 2.0 (More sensitive to initial movement)
                                    active_ids.add(track_id)
                                    is_active = True
                                else:
                                    is_active = False # Still static, ignore
                            else:
                                is_active = False # Determining...
                    else:
                        is_active = True # No ID, assume active (risky but rare)

                    if not is_active:
                        ignored_clumps.append({'box': xyxy, 'reason': 'Static/Pending'})
                        continue

                    weight_clumps.append({'box': xyxy, 'center': (center_x, center_y)})

        # 3. Candidate Selection
        detected_mid_point = None
        valid_clumps = []
        current_scale = 0.0

        if len(weight_clumps) >= 2:
            for wc in weight_clumps:
                area = (wc['box'][2] - wc['box'][0]) * (wc['box'][3] - wc['box'][1])
                wc['score'] = area
                
                # [Fix] Removed Center Priority from individual candidates (it penalized plates at edges)
                
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
            max_pair_score = -float('inf') # Use score instead of just min_y_diff

            import itertools
            for c1, c2 in itertools.combinations(candidates, 2):
                if abs(c1['center'][0] - c2['center'][0]) < width * 0.1:
                    continue
                y_diff = abs(c1['center'][1] - c2['center'][1])
                if y_diff > height * MAX_Y_DIFF_RATIO:
                    continue

                # [New Logic] Pair Selection Score
                # 1. Minimize Y Diff (Alignment)
                # 2. Minimize Distance to Screen Center (X)
                
                mid_x = (c1['center'][0] + c2['center'][0]) / 2
                dist_from_center_x = abs(mid_x - width / 2)
                
                # Score formula: Higher is better
                # Base score can be arbitrary, we penalize deviations
                # We prioritize Y alignment heavily, then center alignment
                
                pair_score = - (y_diff * 2.0 + dist_from_center_x * 0.5)
                
                if pair_score > max_pair_score:
                    max_pair_score = pair_score
                    pair = sorted([c1, c2], key=lambda x: x['center'][0])
                    best_pair = pair
            
            if best_pair:
                valid_clumps = best_pair
                c1 = valid_clumps[0]['center']
                c2 = valid_clumps[1]['center']
                detected_mid_point = (int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2))

                # --- Scale Calculation ---
                w1 = valid_clumps[0]['box'][2] - valid_clumps[0]['box'][0]
                w2 = valid_clumps[1]['box'][2] - valid_clumps[1]['box'][0]
                avg_plate_width_px = (w1 + w2) / 2.0
                
                if avg_plate_width_px > 0:
                    current_scale = PLATE_DIAMETER_M / avg_plate_width_px

        if current_scale > 0:
            scale_factor_history.append(current_scale)

        # 4. Kalman Filter Update / Predict
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

        # --- Draw ---
        if len(all_keypoints) > 0:
            for x, y, conf in all_keypoints:
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1) 
        
        for w in wrist_coords:
             cv2.circle(frame, (int(w[0]), int(w[1])), 6, (255, 0, 0), -1)

        for clump in valid_clumps:
            b = clump['box']
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

        for i in range(1, len(trajectory_points)):
            if trajectory_points[i-1] is not None and trajectory_points[i] is not None:
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)
        
        if final_point:
            cv2.circle(frame, final_point, 8, (0, 0, 255), -1)
        
        out.write(frame)

    cap.release()
    out.release()

    # --- Pass 2: Global Median Scale & Post-Processing ---
    print("\n--- Post-Processing Analysis ---")
    
    # 1. Calculate Global Median Scale
    if not scale_factor_history:
        print("Warning: No valid scale samples found. Defaulting to 1.0.")
        global_scale = 1.0
    else:
        global_scale = np.median(scale_factor_history)
        print(f"Global Median Scale Applied: 1 px = {global_scale:.6f} meters")

    # 2. Extract Y-coords
    y_coords = []
    for p in trajectory_coords:
        if p is not None:
            y_coords.append(p[1])
        else:
            y_coords.append(None) # 欠損保持

    valid_y_for_smooth = [y for y in y_coords if y is not None]
    if not valid_y_for_smooth:
        print("No trajectory data available.")
        return

    # 3. Smoothing (Apply S-G Filter to Y-coords)
    # 欠損値補間 (線形補間) してからスムージング
    # PandasがないのでNumpyでやる
    y_indices = np.arange(len(y_coords))
    y_values = np.array(y_coords, dtype=np.float64)
    
    # None (np.nanに変換)
    y_values_float = []
    for val in y_coords:
        if val is None:
            y_values_float.append(np.nan)
        else:
            y_values_float.append(val)
    y_values_float = np.array(y_values_float)

    # 欠損マスク
    mask = np.isnan(y_values_float)
    # 線形補間
    y_values_float[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_values_float[~mask])
    
    # S-G Filter (Y座標のスムージング)
    smoothed_y = apply_savgol_filter(y_values_float, window_length=15, polyorder=3)

    # 4. Calculate Velocity based on Global Scale & Smoothed Y
    # V = (Y[t] - Y[t-1]) * scale * fps
    velocities_m_s = []
    for i in range(1, len(smoothed_y)):
        dy_px = smoothed_y[i] - smoothed_y[i-1] # 下降が正
        
        # 距離 (絶対値ではなく、変位を見るべきだが、ここでは一般的な「速さ」として絶対値をとるか、
        # あるいは符号付き速度にするか。グラフ化等なら符号ありがいいが、最大速度は絶対値。
        # ここでは絶対値(速さ)を計算
        dist_px = np.abs(dy_px) # 上下動を含めた移動距離
        dist_m = dist_px * global_scale
        v = dist_m * fps
        velocities_m_s.append(v)
    
    # Velocity Smoothing (速度グラフのノイズ除去)
    velocities_m_s = apply_savgol_filter(velocities_m_s, window_length=9, polyorder=2)

    # 5. Count Reps    # State Machine Counting
    rep_count, reps_data = count_reps_state_machine(smoothed_y, velocities_m_s, fps, PLATE_DIAMETER_M)
    
    # Export NPY if requested
    if export_npy:
        try:
            print(f"Exporting velocity data to {export_npy}...")
            np.save(export_npy, velocities_m_s)
        except Exception as e:
            print(f"Failed to export NPY: {e}")

    print(f"Total Reps Detected: {rep_count}")
    for rep in reps_data:
        print(f"Rep {rep['rep_no']}: Peak Velocity = {rep['max_velocity']:.2f} m/s, Avg Velocity = {rep['avg_velocity']:.2f} m/s")

    print(f"Analysis complete. Output video: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze lift with Global Median Scale & State Machine.")
    parser.add_argument("clump_model_path", type=str, help="Path to the trained weight_clump model (.pt file).")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--conf_threshold", type=float, default=0.15, help="Confidence threshold for clump detection.")
    parser.add_argument("--jump_threshold", type=float, default=100.0, help="Maximum allowed pixel jump per frame.")
    parser.add_argument("--plate_diameter", type=float, default=45.0, help="Diameter of the weight plate in cm.")
    parser.add_argument('--output_path', type=str, default='output/output_video.mp4', help='Path to output video')
    parser.add_argument('--export_npy', type=str, default=None, help='Path to export velocity .npy file')
    args = parser.parse_args()

    # FPS Measurement
    import time
    start_time = time.time()
    
    try:
        # Correct Argument Order: clump_model_path, video_path, output_path, ...
        analyze_lift(args.clump_model_path, args.video_path, args.output_path, 
                     args.conf_threshold, args.jump_threshold, args.plate_diameter, 
                     export_npy=args.export_npy)
    except Exception as e:
        import traceback
        traceback.print_exc()

    end_time = time.time()
    elapsed = end_time - start_time
    # Since we can't easily get total frames from here without opening video again, 
    # we rely on analyze_lift execution. 
    # Note: analyze_lift currently doesn't print FPS. 
    # We can approximate average FPS if we knew frame count. 
    # Let's open video just to get frame count for FPS calc? Lightweight.
    
    try:
        cap = cv2.VideoCapture(args.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Total Time: {elapsed:.2f} s")
        print(f"Average FPS: {total_frames / elapsed:.2f}")
    except:
        print(f"Total Time: {elapsed:.2f} s")

if __name__ == "__main__":
    main()

