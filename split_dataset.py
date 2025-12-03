import os
import glob
import random
import shutil

# --- 設定 ---

# スクリプト自身の絶対パスを取得し、それを基準ディレクトリとする
# これにより、VSCodeの実行ボタン(▶)からでもターミナルからでも正しく動作します
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# スクリプトがある場所を基準に、ソースフォルダを指定
IMAGE_SOURCE_DIR = os.path.join(SCRIPT_DIR, "images/train")
LABEL_SOURCE_DIR = os.path.join(SCRIPT_DIR, "labels/train")

# 出力先のベースフォルダ名（スクリプトと同じ場所に作成）
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_split")

# 学習データ（train）の割合 (例: 0.8 = 80%)
TRAIN_RATIO = 0.8

# --- 設定終わり ---

def create_yolo_dataset_structure(base_dir, train_ratio):
    print("データセットの分割を開始します...")
    print(f"基準フォルダ: {SCRIPT_DIR}")
    print(f"画像ソース: {IMAGE_SOURCE_DIR}")
    
    # 1. 出力先のフォルダ構造を作成
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_lbl_dir = os.path.join(base_dir, "train", "labels")
    valid_img_dir = os.path.join(base_dir, "valid", "images")
    valid_lbl_dir = os.path.join(base_dir, "valid", "labels")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_lbl_dir, exist_ok=True)

    # 2. ソースフォルダから画像ファイルのリストを取得
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(IMAGE_SOURCE_DIR, fmt)))

    if not image_files:
        print(f"エラー: '{IMAGE_SOURCE_DIR}' フォルダに画像が見つかりません。")
        print("CVATからエクスポートした 'images/train' フォルダが存在するか確認してください。")
        return

    # 3. ファイルリストをシャッフル
    random.shuffle(image_files)

    # 4. trainとvalidに振り分ける
    split_index = int(len(image_files) * train_ratio)
    train_images = image_files[:split_index]
    valid_images = image_files[split_index:]

    print(f"合計画像数: {len(image_files)}")
    print(f"学習用 (train): {len(train_images)} 枚")
    print(f"検証用 (valid): {len(valid_images)} 枚")

    # 5. ファイルを対応するフォルダにコピー
    def copy_files(file_list, img_dest_dir, lbl_dest_dir):
        for img_path in file_list:
            # 画像ファイル名から、対応するラベルファイル名を生成
            base_filename = os.path.basename(img_path)
            name_part, ext = os.path.splitext(base_filename)
            label_filename = f"{name_part}.txt"
            label_path = os.path.join(LABEL_SOURCE_DIR, label_filename)

            # 画像とラベルをコピー
            shutil.copy(img_path, img_dest_dir)
            
            if os.path.exists(label_path):
                shutil.copy(label_path, lbl_dest_dir)
            else:
                print(f"警告: {label_path} が見つかりません（画像 {base_filename} に対応するラベルがありません）")

    print("\n学習用データをコピー中...")
    copy_files(train_images, train_img_dir, train_lbl_dir)
    
    print("検証用データをコピー中...")
    copy_files(valid_images, valid_img_dir, valid_lbl_dir)

    print("\nデータセットの分割が完了しました。")
    print(f"出力先: '{OUTPUT_DIR}' フォルダ")

if __name__ == "__main__":
    create_yolo_dataset_structure(OUTPUT_DIR, TRAIN_RATIO)