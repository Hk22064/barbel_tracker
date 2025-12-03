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

def get_lifter_info(person_keypoints):
    """リフターの中心点と手首の座標を取得する"""
    # Keypoint indices for YOLOv8-pose:
    # 5: left_shoulder, 6: right_shoulder, 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    
    shoulders = person_keypoints[[5, 6]]
    hips = person_keypoints[[11, 12]]
    wrists = person_keypoints[[9, 10]]
    
    # 可視化されているキーポイントのみを対象 (conf > 0)
    visible_shoulders = shoulders[shoulders[:, 2] > 0]
    visible_hips = hips[hips[:, 2] > 0]
    visible_wrists = wrists[wrists[:, 2] > 0] # 手首
    
    lifter_center = None
    if len(visible_shoulders) > 0 and len(visible_hips) > 0:
        shoulder_mid = np.mean(visible_shoulders[:, :2], axis=0)
        hip_mid = np.mean(visible_hips[:, :2], axis=0)
        lifter_center = (shoulder_mid + hip_mid) / 2
        lifter_center = lifter_center.astype(int)

    # 手首の座標リスト (可視化されているものだけ)
    wrist_coords = []
    if len(visible_wrists) > 0:
        for w in visible_wrists:
            wrist_coords.append(w[:2]) # (x, y)

    return lifter_center, wrist_coords

def main(clump_model_path, video_path, output_path=None, conf_threshold=0.25, jump_threshold=100):
    # 出力フォルダの作成 (outputフォルダがない場合は作成)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        filename = f"output_{base_name}_filtered.mp4"
        output_path = os.path.join(output_dir, filename)
        print(f"Output path not specified. Automatically set to: {output_path}")

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
    trajectory_points = deque(maxlen=128)
    
    last_valid_clumps = None
    last_valid_midpoint = None
    frames_since_last_valid = 0
    INTERPOLATION_FRAME_LIMIT = 10 

    # --- フィルタ設定 ---
    track_history = {} 
    STATIC_VAR_THRESHOLD = 5.0 
    STATIC_HISTORY_LEN = 15    # 少し短くして反応を早くする
    EDGE_MARGIN_RATIO = 0.05   
    MAX_Y_DIFF_RATIO = 0.15    # [New] 左右のプレートの高さのズレ許容率 (画面高さの15%)

    print(f"Starting analysis with JUMP_THRESHOLD = {jump_threshold}, STATIC_FILTER = ON")

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
                    
                    # [Filter 1] 画面端フィルタ
                    if center_x < width * EDGE_MARGIN_RATIO or center_x > width * (1 - EDGE_MARGIN_RATIO):
                        ignored_clumps.append({'box': xyxy, 'reason': 'Edge'})
                        continue

                    # [Filter 2] 静止物体(分散)フィルタ
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
                            # [改善] 履歴が溜まるまでは「疑わしい」として静止扱いにする (初期誤検知対策)
                            # ただし、これをやると最初の15フレーム検出されないので、
                            # 確実に動くバーベルなら判定を甘くしてもいいが、誤検知優先ならTrue
                            is_static = True 
                    
                    if is_static:
                        ignored_clumps.append({'box': xyxy, 'reason': 'Static'})
                        continue

                    weight_clumps.append({'box': xyxy, 'center': (center_x, center_y)})

        # --- 3. 候補選定 (優先度ロジック強化) ---
        mid_point = None
        valid_clumps = []
        
        if len(weight_clumps) >= 2:
            # 基本スコア: 面積が大きいほど有利
            for wc in weight_clumps:
                area = (wc['box'][2] - wc['box'][0]) * (wc['box'][3] - wc['box'][1])
                wc['score'] = area

            # [New] 手首近傍ボーナス
            # 手首が見つかっている場合、手首に近いプレートのスコアを大幅アップ
            if wrist_coords:
                for wc in weight_clumps:
                    min_dist_to_wrist = float('inf')
                    for w in wrist_coords:
                        d = np.linalg.norm(np.array(wc['center']) - w)
                        if d < min_dist_to_wrist:
                            min_dist_to_wrist = d
                    
                    # 手首から近い(例えば画面幅の10%以内)ならスコア倍増
                    if min_dist_to_wrist < width * 0.15:
                        wc['score'] *= 3.0 # ボーナス係数

            # [New] リフター中心距離ペナルティ
            # 手首がない場合、リフター中心を使うが、遠すぎるものはペナルティ
            elif lifter_center is not None:
                 for wc in weight_clumps:
                    dist_to_center = np.linalg.norm(np.array(wc['center']) - lifter_center)
                    # 近ければスコア維持、遠ければ下がる
                    wc['score'] /= (1 + dist_to_center * 0.01)

            # スコア順にソートして上位候補を取得
            weight_clumps.sort(key=lambda wc: wc['score'], reverse=True)
            candidates = weight_clumps[:4] # 上位4つからペアを探す
            
            # --- [New Filter 3] ペアの水平整合性チェック ---
            # 候補の中から「左右にあって」「高さが揃っている」ベストなペアを探す
            best_pair = []
            min_y_diff = float('inf')

            # 総当たりでペアを探す (候補数が少ないのでOK)
            import itertools
            for c1, c2 in itertools.combinations(candidates, 2):
                # 1. 左右に分かれているか？ (X座標の差がある程度あるか)
                if abs(c1['center'][0] - c2['center'][0]) < width * 0.1:
                    continue # 近すぎる(左右じゃない)
                
                # 2. 高さが揃っているか？ (Y座標の差が許容範囲内か)
                y_diff = abs(c1['center'][1] - c2['center'][1])
                if y_diff > height * MAX_Y_DIFF_RATIO:
                    continue # 高さがズレすぎ (片方が床、片方が頭上など)

                # 条件を満たすペアの中で、Y差分が最も小さい(水平な)ものを採用
                # かつ、合計スコアが高いものを優先したいが、水平さを優先
                if y_diff < min_y_diff:
                    min_y_diff = y_diff
                    # X座標でソートして 左, 右 の順にする
                    pair = sorted([c1, c2], key=lambda x: x['center'][0])
                    best_pair = pair
            
            if best_pair:
                valid_clumps = best_pair

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
        
        # --- 確定 ---
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

        # --- 描画 ---
        if len(all_keypoints) > 0:
            for x, y, conf in all_keypoints:
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1) 
        if lifter_center is not None:
            cv2.circle(frame, tuple(lifter_center), 8, (255, 0, 255), -1)

        # 手首 (青)
        for w in wrist_coords:
             cv2.circle(frame, (int(w[0]), int(w[1])), 6, (255, 0, 0), -1)

        for clump in ignored_clumps:
            b = clump['box']
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (128, 128, 128), 1)

        for clump in valid_clumps:
            b = clump['box']
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        
        for i in range(1, len(trajectory_points)):
            if trajectory_points[i-1] is not None and trajectory_points[i] is not None:
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)
        
        out.write(frame)

    # --- 分析結果 ---
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