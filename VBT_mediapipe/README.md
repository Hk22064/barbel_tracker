# VBT Analyzer アプリケーション実行手順

本アプリケーションは、単眼カメラを使用してスクワットの挙上速度（VBT）を計測するプロトタイプです。

## 1. 事前準備 (環境構築)

まず、Anaconda環境を有効化し、必要なライブラリをインストールしてください。

```powershell
# 1. Conda環境の有効化
conda activate yolo

# 2. 依存ライブラリのインストール
pip install mediapipe opencv-python numpy matplotlib
```

## 2. アプリケーションの起動

### Webカメラを使用する場合 (デフォルト)
PCに接続されたWebカメラを使ってリアルタイムで計測します。

```powershell
python app.py
```

### 動画ファイルを使用する場合
既存の動画ファイルを読み込んで解析します。

```powershell
python app.py --video .video/mybench.mp4
python app.py --video ..\video\vertical\front_10rep.mp4
python app.py --video ..\video\horizontal\right_9rep.mp4
```
※ `data/squat.mp4` は任意の動画パスに書き換えてください。


※ `data/squat.mp4` は任意の動画パスに書き換えてください。

### 動画の前処理（推奨）
被写体が遠くに映っている場合、精度が低下することがあります。
その場合、以下のツールで「人を自動で切り抜いた動画」を作成してから解析すると、精度が劇的に向上します。

```powershell
# 動画を自動クロップ（_cropped.mp4 が生成されます）
python preprocess_crop_video.py video/horizontal/front_9rep.mp4

# 生成された動画で解析
python app.py --video video/horizontal/front_9rep_cropped.mp4
```

### ハイブリッドモード (YOLO検出 + MediaPipe姿勢推定)
YOLOで人物を検出し、その領域をクロップしてからMediaPipeで詳細な姿勢推定を行うモードです。
遠距離や複雑な背景で精度が向上します（卒論実験用）。

最新の **YOLO11n** と **Auto-Crop** を組み合わせた専用アプリを用意しました。

```powershell
# Webカメラで実行
python app_yolo11.py

# 動画ファイルで実行
python app_yolo11.py --video video\horizontal\front_9rep.mp4
```

必要なライブラリの追加インストール:
```bash
pip install ultralytics
```

## 3. 操作方法

アプリケーション起動中、以下のキー操作が可能です。

| キー | 機能 | 説明 |
| :---: | :--- | :--- |
| **C** | キャリブレーション | **重要:** バーベルを担いだ状態で直立し、このキーを押してください。<br>左右の手首間距離を「81cm」と仮定して、ピクセル→メートルの変換係数を算出します。 |
| **Q** | 終了 | アプリケーションを終了します。 |

## 4. 画面の見方

- **スケルトン表示**: 推定された体の骨格が表示されます。
- **Velocity**: 現在の挙上速度 (m/s) です。
- **Color Indicator**: 疲労度に応じて文字色が変化します。
  - **緑**: 速度低下なし (<10%)
  - **黄**: 注意 (10-20% 低下)
  - **赤**: 疲労 (>20% 低下)
- **Graph**: 画角右下にリアルタイムの速度グラフが表示されます。

提案手法だけテストするスクリプト作って調整していいよ

この機構実装する前にいつでも戻せるようにしてそれ進めて
テストして全部の動画に対してレップ数が正しくなるようになるまで調整して

実験ごとにexperiment_results内にフォルダ作ってその中に結果入れてくんね？わかりづらい

処理速度とグラフを作成するスクリプト作成して
実行するのはvideoフォルダに入ってる動画

ちょっと使わないスクリプトとか古いバージョンのスクリプト整理してほしいんだけど