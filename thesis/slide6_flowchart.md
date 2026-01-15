# スライド6: システムフロー図

提案システムの処理フローを可視化した図です。
PowerPointなどで図を作成する際の参考にしてください。

## Mermaid形式（対応エディタで図として表示可能）

```mermaid
flowchart TD
    Input[入力映像<br>(Input Video)] --> YOLO[YOLOv11<br>人物領域検出<br>(Person Detection)]
    YOLO -->|Bounding Box| Crop[動的クロッピング<br>Dynamic Lock-Crop]
    Crop -->|Margin +30%| MP[MediaPipe Pose<br>姿勢推定<br>(Keypoint Extraction)]
    MP -->|Local Coordinates| Trans[座標変換<br>Local to Global]
    Trans --> Output[VBT解析<br>(速度算出・Rep検知)]

    style Input fill:#fff,stroke:#333
    style YOLO fill:#bbf,stroke:#333
    style Crop fill:#bfb,stroke:#333
    style MP fill:#f9f,stroke:#333
    style Trans fill:#fdb,stroke:#333
    style Output fill:#ddd,stroke:#333,stroke-width:2px
```

<br>

## テキスト形式（構成案）

PowerPointで作成する場合は、以下のようなブロックを矢印で繋ぐと分かりやすくなります。

1. **[ 入力映像 ]**
   - ↓（矢印）
2. **[ YOLOv11: 人物領域検出 ]**
   - *（説明: リフターの大まかな位置を特定）*
   - ↓ Bounding Box
3. **[ 動的クロッピング ]**
   - *（説明: 検出領域に30%のマージンを加えて切り出し）*
   - ↓ Cropped Image
4. **[ MediaPipe Pose ]**
   - *（説明: 高解像度化された画像から関節点を抽出）*
   - ↓ Local Part Coordinates
5. **[ 座標変換 (Local → Global) ]**
   - *（説明: 全体座標に戻す）*
   - ↓ Global Coordinates
6. **[ VBT解析 ]**
   - *（説明: 速度算出、平滑化、レップカウント）*

---

### 図のポイント
- **ハイブリッドな構造**（YOLO → MediaPipe）が一目で分かるように、YOLOとMediaPipeのブロックを強調（色を変えるなど）すると良いです。
- **「動的クロッピング」**の部分が、本システムの工夫点（精度向上のカギ）なので、ここを少し目立たせるとアピールになります。
