# VBT Analyzer アプリケーション実行手順

本アプリケーションは、単眼カメラを使用してスクワットの挙上速度（VBT）を計測するプロトタイプです。

## 1. 事前準備 (環境構築)

まず、Anaconda環境を有効化し、必要なライブラリをインストールしてください。

```powershell
# 1. Conda環境の有効化
conda activate yolo

# 2. 依存ライブラリのインストール
pip install mediapipe opencv-python numpy matplotlib ultralytics
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
python app.py --video video/squat.mp4
python app.py --video video/vertical/front_10rep.mp4
```

### ハイブリッドモード (YOLO検出 + MediaPipe姿勢推定)
YOLOで人物を検出し、その領域をクロップしてからMediaPipeで詳細な姿勢推定を行うモードです。
遠距離や複雑な背景で精度が向上します（卒論実験用）。

```powershell
# Webカメラで実行
python app_yolo11.py

# 動画ファイルで実行
python app_yolo11.py --video video\horizontal\front_9rep.mp4
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

---

## 5. ファイル構成と役割 (2025/12/19 更新)

このディレクトリ内のスクリプトは、異なる「姿勢推定アルゴリズム」を使用しています。

### アプリケーション (GUI)
| ファイル名 | 使用Estimator | 説明 |
|:-----------|:--------------|:-----|
| `app.py` | `PoseEstimator` | **従来版 (純MediaPipe)**。YOLO不使用。近距離・正面撮影用。<br>シンプルな構成で動作が軽快ですが、遠距離や横向きに弱いです。 |
| `app_yolo11.py` | `Yolo11MediaPipeEstimator` | **提案手法 (Hybrid)**。YOLOv11で人物検出 → クロップ → MediaPipe。<br>遠距離・横向き・複数人環境でもロバストに動作します。 |
| `app_mediapipe.py` | `Yolo11MediaPipeEstimator` | `app_yolo11.py` と同一（リネーム版）。 |

### Estimator クラス (姿勢推定ロジック)
| ファイル名 | クラス名 | アルゴリズム詳細 |
|:-----------|:---------|:-----------------|
| `yolo11_mediapipe_estimator.py` | `Yolo11MediaPipeEstimator` | **提案手法**。YOLOで初期位置をロック(Immediate Lock)し、その領域内でMediaPipeを回します。 |
| `comp_object_estimator.py` | `CompObjectEstimator` | **比較手法B**。物体検出(YOLO)のみでプレートを追跡します。姿勢推定は行いません。 |
| `pose_estimator.py` | `PoseEstimator` | MediaPipe Poseの標準ラッパー。`app.py`で使用。 |
| `yolo_pose_hybrid_estimator_backup.py` | `YoloPoseHybridEstimator` | **比較手法A**。YOLOv8-Pose (Keypoint Detection) を使用する旧ファイル。 |

### テスト・分析スクリプト (CLI)
| ファイル名 | 親ディレクトリの対応スクリプト | 説明 |
|:-----------|:-------------------------------|:-----|
| `../test_proposed_method.py` | - | **提案手法のみ**を全動画でバッチテストします。FPSとRep数を計測。 |
| `../run_batch_analysis.py` | - | 提案手法 vs 比較手法B の比較実験を行います。 |
| `../run_thesis_experiment.py` | - | 卒論用の全条件（3手法）網羅実験スクリプト。 |

---
**注意:** `test_proposed_method.py` は、縦向き動画(`front_10rep`など)に対して自動回転処理を行いますが、`app_yolo11.py` は行いません。そのため、同じ動画でも挙動が異なる場合があります。