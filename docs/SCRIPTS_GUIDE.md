# VBT スクリプト一覧と依存関係

## 概要

このプロジェクトには複数のアプリケーションスクリプトとテストスクリプトがあり、それぞれ異なる「Estimator（姿勢推定クラス）」を使用しています。

---

## Estimator（姿勢推定クラス）一覧

| クラス名 | ファイル | 説明 |
|:---------|:---------|:-----|
| `PoseEstimator` | `pose_estimator.py` | **純粋なMediaPipe**。YOLO不使用。近距離・正面で安定動作。 |
| `Yolo11MediaPipeEstimator` | `yolo11_mediapipe_estimator.py` | **提案手法（Hybrid）**。YOLOで人物検出→クロップ→MediaPipe。遠距離に強い。 |
| `YoloPoseHybridEstimator` | `yolo_pose_hybrid_estimator.py` | **比較手法A**。YOLO11x-Poseのキーポイントを使用。 |
| `CompObjectEstimator` | `comp_object_estimator.py` | **比較手法B**。プレート検出によるバーベル追跡。 |

---

## アプリケーションスクリプト（GUIあり）

| スクリプト | 使用Estimator | 説明 |
|:-----------|:--------------|:-----|
| `VBT_mediapipe/app.py` | `PoseEstimator` | 基本版。純粋なMediaPipeのみ使用。 |
| `VBT_mediapipe/app_yolo11.py` | `Yolo11MediaPipeEstimator` | **提案手法**のGUIアプリ。 |
| `VBT_mediapipe/app_mediapipe.py` | `Yolo11MediaPipeEstimator` | app_yolo11.pyと同等（リネーム版）。 |

---

## テスト・実験スクリプト（CUI）

| スクリプト | 使用Estimator | 説明 |
|:-----------|:--------------|:-----|
| `test_proposed_method.py` | `Yolo11MediaPipeEstimator` | **提案手法のみ**をテスト。全動画をバッチ処理。 |
| `run_batch_analysis.py` | `Yolo11MediaPipeEstimator` + `CompObjectEstimator` | 提案手法と比較手法Bを同時テスト。 |
| `run_thesis_experiment.py` | 全Estimator | 論文用の包括実験スクリプト。 |
| `save_demo_video.py` | `Yolo11MediaPipeEstimator` | デモ動画生成用。 |

---

## ⚠️ 重要な違い

### `app.py` vs `test_proposed_method.py`

| 項目 | `app.py` | `test_proposed_method.py` |
|:-----|:---------|:--------------------------|
| Estimator | `PoseEstimator` (純MediaPipe) | `Yolo11MediaPipeEstimator` (Hybrid) |
| 回転処理 | なし | あり（縦動画用） |
| 用途 | 近距離・正面での使用 | 遠距離・様々な角度のバッチテスト |

**→ `app.py` で5回検出できる動画が `test_proposed_method.py` で0回になる場合、Hybrid版のLock-Crop機構が動画に合っていない可能性があります。**

---

## 推奨コマンド

```bash
# 提案手法（Hybrid）でGUIテスト
python VBT_mediapipe/app_yolo11.py --video video/vertical/front_5rep.mp4

# 純MediaPipeでGUIテスト（従来版）
python VBT_mediapipe/app.py --video video/vertical/front_5rep.mp4

# 提案手法のみバッチテスト
python test_proposed_method.py

# 提案手法 vs 比較手法 バッチテスト
python run_batch_analysis.py
```
