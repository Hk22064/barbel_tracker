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

このディレクトリ内のスクリプトは、役割分担（推定・分析・表示）が明確化されています。

### アプリケーション (GUI / Entry Point)
| ファイル名 | 役割 | 説明 |
|:-----------|:-----|:-----|
| `app_mediapipe.py` | UI・表示 | **現在のメインアプリ**。`Yolo11MediaPipeEstimator` と `VBTAnalyzer` を統合し、映像表示、グラフ描画、ユーザー操作を受け持ちます。 |
| `app.py` | レガシー | 旧バージョン (純MediaPipe)。YOLOを使用せず、近距離・正面撮影に特化しています。 |

### コアロジック (Back-end Classes)
| ファイル名 | クラス名 | 役割・機能 |
|:-----------|:---------|:-----------|
| `yolo11_mediapipe_estimator.py` | `Yolo11MediaPipeEstimator` | **目 (Vision)**。ハイブリッド推定を行います。<br>1. **Detection**: YOLOで初期位置やロスト時の再捜索を行う。<br>2. **Tracking**: MediaPipeでクロップ領域を高精度に追跡する（Auto-Relock機能付き）。<br>3. **Transform**: 局所座標を全体座標に変換する。 |
| `vbt_analyzer.py` | `VBTAnalyzer` | **頭脳 (Logic)**。物理計算と分析を行います。<br>1. **Physics**: 平滑化フィルタを用いた速度計算。<br>2. **State Machine**: 挙上・下降・静止の状態遷移とレップ判定。<br>3. **Calibration**: 手首間距離(81cm)や体幹長に基づくスケール計算。<br>4. **Analysis**: 疲労度やドロップオフ率の算出。 |

### その他のEstimator (比較実験用)
| ファイル名 | クラス名 | 説明 |
|:-----------|:---------|:-----|
| `comp_object_estimator.py` | `CompObjectEstimator` | **比較手法B**。物体検出(YOLO)のみでプレートを追跡。姿勢推定なし。 |
| `pose_estimator.py` | `PoseEstimator` | 標準MediaPipeラッパー。`app.py`で使用。 |

### テスト・分析スクリプト (CLI)
| ファイル名 | 親フォルダのスクリプト | 説明 |
|:-----------|:-----------------------|:-----|
| `../test_proposed_method.py` | - | **提案手法のバッチテスト**。全動画ファイルに対して `Yolo11MediaPipeEstimator` + `VBTAnalyzer` を走らせ、FPSとRep数を計測します。 |

---

## 6. 精度調整パラメーター (Tuning Guide)

使用環境や被写体に合わせて、以下の変数を調整することで精度を最適化できます。

### 1. 姿勢推定 (Vision) : `yolo11_mediapipe_estimator.py`
| 変数名 | 現在値 | 説明 | 変更時の影響 |
|:---|:---|:---|:---|
| `margin_x`, `margin_y` | `0.3` (30%) | クロップ枠の余白率 | **大**: 手先が見切れにくくなるが、被写体が小さくなりジッターノイズが増える。<br>**小**: 画素密度が上がり精度が良くなるが、腕を伸ばした時に見切れて手首が飛びやすくなる。 |
| `lost_tracking_threshold` | `30` | 再ロックまでの猶予フレーム | **大**: 人が一瞬隠れてもロックを継続する。<br>**小**: ロスト時に素早くYOLO復帰するが、誤検出で頻繁にリセットされやすくなる。 |

### 2. 数値分析 (Analyzer) : `vbt_analyzer.py`
| 変数名 | 現在値 | 説明 | 変更時の影響 |
|:---|:---|:---|:---|
| `smoothing_window` | `5` | 移動平均バッファサイズ | **大**: グラフが滑らかになるが、瞬発的な最大速度が低く出る（鈍る）。<br>**小**: 反応が鋭敏になるが、少しのノイズでグラフがギザギザになる。 |
| `velocity_threshold` | `0.05` | 静止判定の速度閾値 (m/s) | これ以下の速度は「静止」とみなすノイズフィルタ。大きくしすぎるとゆっくりな動作（Stick）を検出できない。 |
| `real_distance_m` | `0.81` | 手首間距離 (m) | **キャリブレーションの基準値**。実際の被験者のグリップ幅に合わせて正確に入力すると、速度計測の絶対精度が向上する。 |

