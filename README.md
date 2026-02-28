# NDLOCR-Lite PDF OCR Pipeline

PDFを入力として受け取り、NDLOCR-Lite CLI経由で日本語OCRを実行し、テキストを出力するパイプライン。日本語の文節を考慮した改行整形機能を備えています。

## 概要

```
PDF → (テキスト埋め込み検出 → スキップ判定)
    → PyMuPDF（ページごとPNG、DPI 300）
    → バッチ分割（デフォルト20ページ/バッチ）
    → ndlocr-lite CLI（バッチごと並列実行）
    → 後処理（誤認識修正、ページ区切り追加）
    → 日本語文節単位の改行整形
    → 出力
```

### 主な特徴

- **GPU不要・完全ローカル動作**
- **バッチ処理**: 大きなPDFも分割して処理（タイムアウト回避）
- **高精度OCR**: DPI 300で高品質な画像変換
- **後処理機能**:
  - 典型的な誤認識の自動修正（武内党→武内覚など）
  - ページ区切りの自動挿入
  - **BudouXによる日本語文節単位の改行整形**（Google製MLベース）
  - 固有名詞の保持（Linuxエンジニア、プログラミングなど）

## インストール

### 前提条件

- Python 3.11+
- uv（推奨）または pip

### 1. ndlocr-lite のインストール

```bash
git clone https://github.com/ndl-lab/ndlocr-lite
cd ndlocr-lite
uv tool install .
```

インストール確認:
```bash
ndlocr-lite --help
```

### 2. Python依存パッケージのインストール

```bash
pip install pymupdf budoux
```

または:
```bash
pip install -r requirements.txt
```

## ファイル構成

```
project/
├── README.md                  # このファイル
├── pdf_to_text.py            # メインスクリプト
├── pdf_to_pages.py           # PDF→PNG変換モジュール
├── requirements.txt          # 依存パッケージ
├── input/                    # 処理対象PDFを置くディレクトリ
└── output/                   # OCR結果テキストの出力先
```

## 使い方

### 基本的な使い方

#### 単一PDFの処理

```bash
python pdf_to_text.py input/document.pdf output/
```

#### ディレクトリ内のPDFを一括処理

```bash
python pdf_to_text.py input/ output/
```

### 高度なオプション

#### DPIの変更（精度 vs 速度）

```bash
# 高精度（処理は遅い）
python pdf_to_text.py input.pdf output/ --dpi 400

# 標準（デフォルト: 300）
python pdf_to_text.py input.pdf output/ --dpi 300

# 高速（処理は速いが精度低下）
python pdf_to_text.py input.pdf output/ --dpi 200
```

#### バッチサイズの調整

ページ数が多いPDFは小さいバッチに分割して処理します:

```bash
# 小さいバッチ（10ページ/バッチ）- タイムアウト回避向け
python pdf_to_text.py input.pdf output/ --batch-size 10

# 大きいバッチ（50ページ/バッチ）- 高速化向け
python pdf_to_text.py input.pdf output/ --batch-size 50
```

#### 並列処理（複数PDFの同時処理）

```bash
# 4ワーカーで並列処理
python pdf_to_text.py input/ output/ --workers 4
```

#### テキスト埋め込み済みPDFの強制OCR

通常はテキスト埋め込み済みPDFはスキップされますが、強制的にOCRできます:

```bash
python pdf_to_text.py input.pdf output/ --no-skip-embedded
```

## 後処理の詳細

### 1. 誤認識の自動修正

OCR結果に対して以下の誤認識を自動修正します:

| 誤認識 | 修正後 |
|--------|--------|
| 武内党 | 武内覚 |
| Iinux | Linux |
| ウェプ | ウェブ |
| 202年 | 2022年 |
| TM、、 | ™、®、© |

### 2. ページ区切り

各ページの先頭に `--- Page X ---` 形式の区切りを挿入:

```
--- Page 1 ---
タイトルページの内容...

--- Page 2 ---
第一章の内容...
```

### 3. 日本語文節単位の改行整形（BudouX）

**BudouX**（Google製）を使用し、機械学習ベースの文節分割で自然な改行位置を決定：

**改善前:**
```
この本の構成は、各章の前半は一般的なもので、後半は最新の情報となってい
ます。
```

**改善後:**
```
この本の構成は、各章の前半は一般的なもので、後半は最新の情報となっています。
```

### 4. 固有名詞の保持

以下のような固有名詞は分割されません:

- Linuxエンジニア
- プログラミング
- コンテナ技術
- 子プロセス
- オペレーティングシステム

**改善前:**
```
システム設計やプログ
ラミングができます。
```

**改善後:**
```
システム設計やプログラミングができます。
```

## 設定可能なパラメータ

`pdf_to_text.py` の先頭で以下の定数を変更できます:

```python
# --- 設定 ---
DEFAULT_DPI = 300              # デフォルトの解像度
BATCH_SIZE = 20                # 1バッチあたりのページ数
OCR_TIMEOUT_SEC = 300          # 1バッチあたりのタイムアウト（秒）
DEFAULT_MAX_LINE_LENGTH = 70   # 1行の目標文字数
```

### 1行の文字数を変更する場合

```python
# 短い行（モバイル向け）
DEFAULT_MAX_LINE_LENGTH = 50

# 標準
DEFAULT_MAX_LINE_LENGTH = 70

# 長い行（PC向け）
DEFAULT_MAX_LINE_LENGTH = 100
```

## パラメータ指針

### DPI設定

| DPI | 用途 | 速度 | 推奨シナリオ |
|-----|------|------|-------------|
| 150 | 大量処理・速度優先 | 速い | 下書き・草稿 |
| 200 | 速度優先 | 普通 | 試行・テスト |
| 300 | **標準** | やや遅い | 通常用途 |
| 400 | 高精度 | 遅い | 細かい文字・品質重視 |

### バッチサイズ

| サイズ | 用途 | 備考 |
|--------|------|------|
| 10 | タイムアウト回避 | 大きいPDF・画質が悪い場合向け |
| 20 | **標準** | バランスの取れた設定 |
| 50 | 高速化 | 小さいPDF・高スペック環境向け |

### 1行の文字数

| 文字数 | 用途 | 備考 |
|--------|------|------|
| 50 | モバイル表示向け | スマートフォンで読みやすい |
| 70 | **標準** | 一般的な文書向け |
| 100 | PC表示向け | デスクトップでの閲覧向け |
| 120 | ワイド画面向け | 横長のディスプレイ向け |

## トラブルシューティング

### `ndlocr-lite: command not found`

**解決策:** ndlocr-lite リポジトリ内で `uv tool install .` を実行してください。

### OCR結果が空

**解決策:** 
1. DPIを上げる（`--dpi 300` または `--dpi 400`）
2. バッチサイズを小さくする（`--batch-size 10`）

### タイムアウトエラー

**解決策:** 
1. バッチサイズを小さくする（`--batch-size 10`）
2. `OCR_TIMEOUT_SEC` の値を増やす（デフォルト: 300秒）

### メモリ不足

**解決策:**
1. `--workers 1`（デフォルト）で逐次処理にする
2. バッチサイズを小さくする
3. DPIを下げる

### テキスト埋め込み済みPDFが誤検出される

**解決策:** 
1. `--no-skip-embedded` で強制OCR
2. `has_embedded_text()` の `threshold` を調整

### 日本語の改行が不自然

**解決策:**
1. `DEFAULT_MAX_LINE_LENGTH` を調整（50-120文字）
2. 固有名詞パターンに該当単語を追加

## ライセンス

このプロジェクトは Apache License 2.0 の下で公開されています。

## 依存関係

- **pymupdf**: PDF→画像変換
- **budoux**: 日本語文節分割（Google製MLベース）
- **ndlocr-lite**: OCRエンジン（別途インストールが必要）

## 注意事項

- **対応PDF**: スキャン画像型PDF。テキスト埋め込み済みPDFは自動検出してテキスト抽出にフォールバック（`--no-skip-embedded` で強制OCRも可）。
- **古典籍・くずし字**: 本パイプラインは現代活字向け。くずし字は `ndl-lab/ndlkotenocr-lite` を使用してください。
- **処理時間**: ノートPCのCPUで1ページ数秒〜十数秒程度（モデルのロード時間が初回にかかる）。
- **CLI互換性**: ndlocr-lite のバージョンによりCLI引数や出力形式が変わる可能性がある。導入時に `ndlocr-lite --help` で確認してください。
