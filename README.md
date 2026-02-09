# Online IBVS Controller

## 1) これは何か
- `isvs.py` でオンラインIBVS制御を実行
- 特徴対応は DINO/CLIP/Ours（`config.py` のシナリオ）を利用
- 実行結果としてフレーム列、CSV、任意で動画を保存

## 2) 前提（OS, GPU/CPU, 主要依存）
- 想定OS: Windows/Linux
- Python: 3.10+
- 主要依存: `torch`, `transformers`, `opencv-contrib-python`, `pandas`, `Pillow`
- ROS送信を使う場合: `roslibpy`
- Baslerカメラを使う場合: `pypylon`

## 3) セットアップ（最短手順）
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 4) 最小実行例（コピペ可能）
オフライン（カメラ/ROSなし）:
```bash
python isvs.py --scenario dino --target data/samples/target.png --mask data/samples/mask.png --observed data/samples/observed.png --output-dir outputs/demo --max-iter 3
```

設定ファイル利用:
```bash
python isvs.py --config configs/default.yaml --target data/samples/target.png --mask data/samples/mask.png
```

## 5) データの置き場所（サンプル、取得方法）
- 最小サンプル: `data/samples/target.png`, `data/samples/observed.png`, `data/samples/mask.png`
- 出力: `--output-dir` 配下
  - `observed_frames/*.png`
  - `valdata.csv`
  - `observed.mp4`（`--write-video` 時）

## 6) よくあるエラーと対処
- `roslibpy is required`
  - `--ros-enabled` 利用時のみ必要。不要ならフラグを外す
- `pypylon is required`
  - `--use-camera` 利用時のみ必要
- `No valid feature rows`
  - マスクが狭すぎる、または特徴抽出が不安定。マスク/シナリオを見直す
- 依存を最小に試したい
  - `--use-dummy-extractor` を付けると軽量特徴で実行できる（スモーク用）

## 7) プロジェクト構造（各ディレクトリの役割）
```text
online_ibvs_controller/
├─ isvs.py                # CLIエントリ（IBVSループ）
├─ config.py              # シナリオ設定
├─ configs/default.yaml   # 実行設定例
├─ src/                   # 特徴抽出・対応・I/O
├─ data/samples/          # 最小サンプル入力
└─ scripts/smoke_test.py  # スモークテスト
```

## 8) ライセンス/引用（third_party含む）
- このディレクトリには外部リポジトリのソース同梱はありません
- 使用ライブラリ・モデルのライセンスは各配布元に従ってください
