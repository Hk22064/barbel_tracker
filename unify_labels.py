

import os
import shutil

# --- 設定 ---

# 入力元と出力先、クラスマッピングのリスト
# 例:
# CONFIGS = [
#     {
#         "SOURCE_LABEL_DIR": "/path/to/source/labels",
#         "DEST_LABEL_DIR": "/path/to/destination/labels",
#         "CLASS_MAPPING": {0: 0, 1: 0} # 元のクラスID -> 新しいクラスID
#     },
#     # ... 他の設定 ...
# ]
CONFIGS = [
    {
        "SOURCE_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/barbell_detection.v1i.yolov11/train/labels",
        "DEST_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/barbell_detection.v1i.yolov11/train/labels_unified",
        "CLASS_MAPPING": {0: 0} 
    },
    {
        "SOURCE_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/barbell_detection.v1i.yolov11/valid/labels",
        "DEST_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/barbell_detection.v1i.yolov11/valid/labels_unified",
        "CLASS_MAPPING": {0: 0}
    },
    {
        "SOURCE_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/WeightPlate.v2i.yolov11/train/labels",
        "DEST_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/WeightPlate.v2i.yolov11/train/labels_unified",
        "CLASS_MAPPING": {0: 0}
    },
    {
        "SOURCE_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/WeightPlate.v2i.yolov11/valid/labels",
        "DEST_LABEL_DIR": "C:/Users/kurau/Bench_pose/new_mylab/subset/external_dataset/WeightPlate.v2i.yolov11/valid/labels_unified",
        "CLASS_MAPPING": {0: 0}
    }
] 

# --- 設定ここまで ---


def unify_labels(source_dir, dest_dir, class_mapping):
    """
    YOLO形式のラベルファイルのクラスIDを変換し、新しいディレクトリに保存する。

    Args:
        source_dir (str): 変換元のラベルファイルが含まれるディレクトリ。
        dest_dir (str): 変換後のラベルファイルを保存するディレクトリ。
        class_mapping (dict): クラスIDの変換ルールを定義した辞書。
                               例: {0: 0, 1: 0}
                               値がNoneのクラスは行ごと削除される。
    """
    if os.path.exists(dest_dir):
        print(f"警告: 出力先ディレクトリ {dest_dir} は既に存在します。中身を一度削除します。")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    print(f"処理中: {source_dir} -> {dest_dir}")

    file_count = 0
    for filename in os.listdir(source_dir):
        if not filename.endswith(".txt"):
            continue

        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        with open(source_path, 'r') as f_in:
            lines = f_in.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                original_class_id = int(parts[0])
                
                if original_class_id in class_mapping:
                    new_class_id = class_mapping[original_class_id]
                    if new_class_id is not None:
                        parts[0] = str(new_class_id)
                        new_lines.append(" ".join(parts) + "\n")
                # マッピングにないクラスIDは無視（行を削除）
                
            except (ValueError, IndexError) as e:
                print(f"警告: ファイル {filename} の行 '{line.strip()}' の処理中にエラー: {e}")
                continue
        
        if new_lines:
            with open(dest_path, 'w') as f_out:
                f_out.writelines(new_lines)
            file_count += 1

    print(f"完了: {file_count} 個のラベルファイルを {dest_dir} に変換・保存しました。")


if __name__ == '__main__':
    if not CONFIGS:
        print("エラー: `CONFIGS`リストが空です。処理するデータセットの情報を設定してください。")
    else:
        for i, config in enumerate(CONFIGS):
            print(f"\n--- 設定 {i+1}/{len(CONFIGS)} を処理します ---")
            source = config.get("SOURCE_LABEL_DIR")
            dest = config.get("DEST_LABEL_DIR")
            mapping = config.get("CLASS_MAPPING")

            if not all([source, dest, mapping is not None]):
                print(f"エラー: 設定 {i+1} に必要なキー（SOURCE_LABEL_DIR, DEST_LABEL_DIR, CLASS_MAPPING）が不足しています。")
                continue
            
            unify_labels(source, dest, mapping)
        print("\n全ての処理が完了しました。")

