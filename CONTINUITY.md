# 継続性台帳 (Continuity Ledger)

## Goal (incl. success criteria):
ローカルの変更をGitHubにプッシュし、リモートリポジトリを最新の状態（src構成）に同期させる。
その後、検証用フォルダで再度プルして動作を確認する。

## Constraints/Assumptions:
- 動作環境: Windows (Powershell)
- リポジトリ: https://github.com/Hk22064/barbel_tracker.git
- 動画ファイルなど大容量データは.gitignoreで除外されていると仮定

## Key decisions:
- コミットメッセージ: "Refactor: Update directory structure to src/ layout and add missing requirements"

## State:

### Done:
- リモートURLの確認
- 検証用フォルダ作成・クローン（旧状態で失敗）
- 検証による問題特定（`src`欠落、`VBT_mediapipe`残存）

### Now:
- ローカル変更のコミットとプッシュ
- 検証用フォルダでの `git pull` と再確認

### Next:
- 最終確認報告

### Next:
- 結果報告

## Open questions (UNCONFIRMED if needed):
- 実行時に特定のハードウェア（Webカメラ等）が必要か？（UNCONFIRMED）

## Working set (files/ids/commands):
- `c:\Users\kurau\Bench_pose\new_mylab\verification_test` (検証用)
- `c:\Users\kurau\Bench_pose\new_mylab\barbel_tracker` (参照元)
