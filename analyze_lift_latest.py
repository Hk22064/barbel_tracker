import cv2
import argparse
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import find_peaks
from collections import deque

def calculate_velocity(coords, fps):
    """各フレーム間の速度を計算する"""
    velocities = []
    if len(coords) < 2:
        return velocities

    for i in range(1, len(coords)):
        p1 = coords[i-1]
        p2 = coords[i]
        if p1 is None or p2 is None:
            velocities.append(0)
            continue
            
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        velocity = distance * fps
        velocities.append(velocity)
        
    return velocities

def count_reps(y_coords, prominence_threshold=10):
    """y座標のリストからレップ数をカウントする"""
    if not y_coords:
        return 0, []
    
    inverted_y = [-y for y in y_coords if y is not None]
    if not inverted_y:
        return 0, []
    
    peaks, _ = find_peaks(inverted_y, prominence=prominence_threshold)
    return len(peaks), peaks

def moving_average(data, window_size=5):
    """移動平均を計算してデータを平滑化する"""
    if len(data) < window_size:
        return []
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def get_lifter_center(person_keypoints):
    """肩と腰のキーポイントからリフターの中心点を計算する"""
    # Keypoint indices for shoulders and hips in YOLOv8-pose
    # 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
    shoulders = person_keypoints[[5, 6]]
    hips = person_keypoints[[11, 12]]
    
    # 可視化されているキーポイントのみを対象
    visible_shoulders = shoulders[shoulders[:, 2] > 0]
    visible_hips = hips[hips[:, 2] > 0]
    
    if len(visible_shoulders) == 0 or len(visible_hips) == 0:
        return None

    shoulder_mid = np.mean(visible_shoulders[:, :2], axis=0)
    hip_mid = np.mean(visible_hips[:, :2], axis=0)
    
    lifter_center = (shoulder_mid + hip_mid) / 2
    return lifter_center.astype(int)

def main(clump_model_path, video_path, output_path=None, conf_threshold=0.25, jump_threshold=100):
    # 出力フォルダの作成 (outputフォルダがない場合は作成)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 出力パスが指定されていない場合、入力ファイル名に基づいて自動生成し、outputフォルダに保存
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        filename = f"output_{base_name}_filtered.mp4"
        output_path = os.path.join(output_dir, filename)
        print(f"Output path not specified. Automatically set to: {output_path}")

    # モデルをロード
    clump_model = YOLO(clump_model_path)
    pose_model = YOLO('yolo11m-pose.pt') # 必要に応じて yolov8n-pose.pt 等に変更

    # 動画をキャプチャ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # 動画のプロパティ
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 動画書き出し用の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trajectory_coords = []
    trajectory_points = deque(maxlen=128)
    
    # --- Interpolation variables ---
    last_valid_clumps = None
    last_valid_midpoint = None
    frames_since_last_valid = 0
    INTERPOLATION_FRAME_LIMIT = 10 

    # --- ★★★ 追加設定: 静止物体フィルタ用変数 ★★★ ---
    track_history = {} # IDごとのY座標履歴: {track_id: deque([y1, y2, ...])}
    STATIC_VAR_THRESHOLD = 5.0 # 分散の閾値 (これ以下なら静止しているとみなす)
    STATIC_HISTORY_LEN = 30    # 何フレーム見て判断するか
    EDGE_MARGIN_RATIO = 0.05   # 画面端5%は無視する

    print(f"Starting analysis with JUMP_THRESHOLD = {jump_threshold}, STATIC_FILTER = ON")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. 姿勢推定 ---
        pose_results = pose_model(frame, verbose=False)
        lifter_center = None
        all_keypoints = []
        if pose_results[0].keypoints and len(pose_results[0].keypoints.data) > 0:
            person_keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
            lifter_center = get_lifter_center(person_keypoints)
            all_keypoints = person_keypoints 

        # --- 2. 物体検出 (Tracking Modeに変更) ---
        # persist=True でIDを維持する
        clump_results = clump_model.track(frame, persist=True, verbose=False, conf=conf_threshold)
        
        weight_clumps = []
        ignored_clumps = [] # デバッグ描画用

        for r in clump_results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: 
                    xyxy = box.xyxy[0].cpu().numpy()
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    
                    # --- [Filter 1] 画面端フィルタ ---
                    # 画面の左右 5% にある物体は無視 (ラック等の可能性が高い)
                    if center_x < width * EDGE_MARGIN_RATIO or center_x > width * (1 - EDGE_MARGIN_RATIO):
                        ignored_clumps.append({'box': xyxy, 'reason': 'Edge'})
                        continue

                    # --- [Filter 2] 静止物体(分散)フィルタ ---
                    is_static = False
                    if box.id is not None:
                        track_id = int(box.id.item())
                        
                        # 履歴を更新
                        if track_id not in track_history:
                            track_history[track_id] = deque(maxlen=STATIC_HISTORY_LEN)
                        track_history[track_id].append(center_y)

                        # 履歴が十分に溜まったら分散をチェック
                        if len(track_history[track_id]) >= STATIC_HISTORY_LEN:
                            y_variance = np.var(track_history[track_id])
                            # 分散が閾値以下 = 動きがない = 背景のプレート
                            if y_variance < STATIC_VAR_THRESHOLD:
                                is_static = True
                    
                    if is_static:
                        ignored_clumps.append({'box': xyxy, 'reason': 'Static'})
                        continue

                    # 有効な候補として追加
                    weight_clumps.append({'box': xyxy, 'center': (center_x, center_y)})

        # --- 3. フィルタリング & 4. 軌道計算 ---
        mid_point = None
        valid_clumps = []
        
        # 候補選定
        if len(weight_clumps) >= 2:
            # 面積順でソート (大きい順)
            weight_clumps.sort(key=lambda wc: (wc['box'][2] - wc['box'][0]) * (wc['box'][3] - wc['box'][1]), reverse=True)
            candidates = weight_clumps[:5] # Top 5
            
            # リフターに近い順でソート
            if lifter_center is not None:
                candidates.sort(key=lambda wc: np.linalg.norm(np.array(wc['center']) - lifter_center))
                valid_clumps = candidates[:2]
            else:
                # リフターが見つからない場合は単純に面積最大の2つ
                valid_clumps = candidates[:2]

        # --- ワープ検知 ---
        is_jump_too_big = False
        if len(valid_clumps) == 2:
            c1 = valid_clumps[0]['center']
            c2 = valid_clumps[1]['center']
            potential_mid = (int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2))

            if last_valid_midpoint is not None:
                dist = np.sqrt((potential_mid[0] - last_valid_midpoint[0])**2 + (potential_mid[1] - last_valid_midpoint[1])**2)
                if dist > jump_threshold:
                    is_jump_too_big = True
                    # print(f"Warning: Detected abnormal jump ({dist:.1f} px). Ignoring this frame.")

        # --- 確定ロジック ---
        if len(valid_clumps) == 2 and not is_jump_too_big:
            center1 = valid_clumps[0]['center']
            center2 = valid_clumps[1]['center']
            mid_point = (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2))
            
            last_valid_clumps = valid_clumps 
            last_valid_midpoint = mid_point
            frames_since_last_valid = 0
            
        elif last_valid_clumps is not None and frames_since_last_valid < INTERPOLATION_FRAME_LIMIT:
            center1 = last_valid_clumps[0]['center']
            center2 = last_valid_clumps[1]['center']
            mid_point = (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2))
            frames_since_last_valid += 1
            valid_clumps = last_valid_clumps
        else:
            last_valid_clumps = None
            last_valid_midpoint = None

        trajectory_coords.append(mid_point)
        if mid_point:
            trajectory_points.append(mid_point)

        # --- 描画処理 ---
        # 1. 姿勢推定 (黄色)
        if len(all_keypoints) > 0:
            for x, y, conf in all_keypoints:
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1) 
        if lifter_center is not None:
            cv2.circle(frame, tuple(lifter_center), 8, (255, 0, 255), -1)

        # 2. 無視されたプレート (グレー枠) - デバッグ用
        for clump in ignored_clumps:
            b = clump['box']
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (128, 128, 128), 1)
            # cv2.putText(frame, clump['reason'], (int(b[0]), int(b[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)

        # 3. 有効なプレート (緑枠)
        for clump in valid_clumps:
            b = clump['box']
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        
        # 4. 軌跡 (赤線)
        for i in range(1, len(trajectory_points)):
            if trajectory_points[i-1] is not None and trajectory_points[i] is not None:
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)
        
        out.write(frame)

    # --- 分析結果のサマリー ---
    y_coords = [p[1] for p in trajectory_coords if p is not None]
    
    if not y_coords:
        print("Could not detect valid trajectory.")
    else:
        smoothed_y = moving_average(y_coords, window_size=5)
        if len(smoothed_y) == 0:
            smoothed_y = y_coords

        num_reps, peak_indices = count_reps(list(smoothed_y), prominence_threshold=15)
        print(f"Total Reps Detected: {num_reps}")

        all_velocities = calculate_velocity(trajectory_coords, fps)
        valid_velocities = [v for p, v in zip(trajectory_coords, [0] + all_velocities) if p is not None]
        
        rep_start_idx = 0
        peak_indices_adjusted = [p + (len(y_coords) - len(smoothed_y)) // 2 for p in peak_indices]

        for i, peak_idx in enumerate(peak_indices_adjusted):
            next_peak_idx = peak_indices_adjusted[i+1] if i + 1 < len(peak_indices_adjusted) else len(y_coords)
            v_slice = valid_velocities[rep_start_idx:next_peak_idx]

            if v_slice:
                avg_velocity = np.mean(v_slice)
                max_velocity = np.max(v_slice)
                print(f"Rep {i+1}: Peak Velocity = {max_velocity:.2f} pixels/sec, Avg Velocity = {avg_velocity:.2f} pixels/sec")
            
            rep_start_idx = peak_idx + 1

    cap.release()
    out.release()
    print(f"Analysis complete. Output video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze lift trajectory using Pose and Object Detection.")
    parser.add_argument("clump_model_path", type=str, help="Path to the trained weight_clump model (.pt file).")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output video.")
    parser.add_argument("--conf_threshold", type=float, default=0.15, help="Confidence threshold for clump detection.")
    parser.add_argument("--jump_threshold", type=float, default=100.0, help="Maximum allowed pixel jump per frame.")
    
    args = parser.parse_args()
    main(args.clump_model_path, args.video_path, args.output_path, args.conf_threshold, args.jump_threshold)