# NDLOCR-Lite PDF OCR Pipeline

## 概要

PDFを入力として受け取り、NDLOCR-Lite CLI経由で日本語OCRを実行し、テキストを出力するパイプライン。
NDLOCR-Lite CLIは画像入力のみ対応のため、PyMuPDFでPDF→PNG変換を前処理として挟む。

```
PDF → (テキスト埋め込み検出 → スキップ判定)
    → PyMuPDF（ページごとPNG）
    → バッチ分割（デフォルト20ページ/バッチ）
    → ndlocr-lite CLI（バッチごと並列実行）
    → テキスト結合
    → 出力
```

GPU不要・完全ローカル動作。

---

## セットアップ

### 前提条件

- Python 3.11+
- uv（推奨）または pip

### 1. ndlocr-lite インストール

```bash
git clone https://github.com/ndl-lab/ndlocr-lite
cd ndlocr-lite
uv tool install .
```

`ndlocr-lite` コマンドが使えることを確認：

```bash
ndlocr-lite --help
```

> **⚠️ CLI引数の確認**
> 本ドキュメントでは `ndlocr-lite --sourcedir <DIR> --output <DIR>` を想定しているが、
> バージョンによって引数が異なる場合がある。必ず `ndlocr-lite --help` で実際の仕様を確認し、
> `pdf_to_text.py` の `subprocess.run` 呼び出しを合わせること。

### 2. Python依存パッケージ インストール

```bash
pip install pymupdf
```

---

## ファイル構成

```
project/
├── AGENTS.md          # このファイル
├── pdf_to_text.py     # メインスクリプト（PDF→テキスト一括処理）
├── pdf_to_pages.py    # PDF→PNG変換モジュール
├── input/             # 処理対象PDFを置くディレクトリ
└── output/            # OCR結果テキストの出力先
```

---

## 実装仕様

### pdf_to_pages.py

PDFを受け取り、ページごとにPNG画像を指定ディレクトリに出力する。
テキスト埋め込み済みPDFの検出機能も提供する。

```python
import fitz  # PyMuPDF
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def has_embedded_text(pdf_path: str, sample_pages: int = 3, threshold: int = 50) -> bool:
    """
    PDFにテキストレイヤーが埋め込まれているか判定する。

    先頭数ページからテキスト抽出を試み、一定文字数以上あれば
    テキスト埋め込み済みと判断する（＝OCR不要）。

    Args:
        pdf_path: 入力PDFのパス
        sample_pages: チェックするページ数（先頭から）
        threshold: テキスト埋め込みありと判定する最小文字数

    Returns:
        True ならテキスト埋め込み済み（OCR不要）
    """
    doc = fitz.open(pdf_path)
    try:
        pages_to_check = min(sample_pages, len(doc))
        total_chars = sum(len(doc[i].get_text().strip()) for i in range(pages_to_check))
        return total_chars >= threshold
    finally:
        doc.close()


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200) -> list[Path]:
    """
    PDFをページごとのPNG画像に変換する。

    Args:
        pdf_path: 入力PDFのパス
        output_dir: PNG出力先ディレクトリ
        dpi: 解像度（デフォルト200。精度を上げたい場合は300）

    Returns:
        生成した画像ファイルのパスリスト（ページ順）
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72dpiがPDFの基準

    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            img_path = output_dir / f"page_{i+1:04d}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
            logger.debug(f"  変換完了: {img_path.name}")
    finally:
        doc.close()

    return image_paths
```

### pdf_to_text.py

メインスクリプト。PDF→PNG変換→ndlocr-lite実行→テキスト結合を一括で行う。

```python
import subprocess
import shutil
import sys
import tempfile
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdf_to_pages import pdf_to_images, has_embedded_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- 設定 ---
NDLOCR_CMD = "ndlocr-lite"
OCR_TIMEOUT_SEC = 300  # 1バッチあたりのタイムアウト（5分）
BATCH_SIZE = 20  # 1回のOCR処理で処理するページ数
DEFAULT_DPI = 300  # デフォルトの解像度（精度向上のため300に設定）


def _check_ndlocr() -> None:
    """ndlocr-lite コマンドが利用可能か事前チェックする。"""
    if shutil.which(NDLOCR_CMD) is None:
        raise FileNotFoundError(
            f"'{NDLOCR_CMD}' が見つかりません。\n"
            "ndlocr-lite リポジトリ内で `uv tool install .` を実行してください。"
        )


def _post_process_text(text: str) -> str:
    """
    OCR結果のテキストに後処理を適用する。
    
    主な処理:
    - 典型的な誤認識の修正
    - 記号の正規化
    - 不要な空白の削除
    """
    import re
    
    # 典型的な誤認識の修正
    corrections = {
        # 英字の誤認識
        'Iinux': 'Linux',
        
        # 人名の誤認識
        '武内党': '武内覚',
        
        # カタカナの誤認識
        'ウェプ': 'ウェブ',
        'ペース': 'ベース',
        
        # 記号の正規化
        'TM、、': '™、®、©',
        
        # 数字の修正
        '202年': '2022年',
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # 連続する空白を1つに
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'　{2,}', '　', text)
    
    return text


def _collect_ocr_texts(ocr_dir: Path, start_page: int = 1) -> tuple[str, int]:
    """
    ndlocr-lite の出力ディレクトリからテキストを収集・結合する。
    ページ区切りを挿入し、後処理を適用する。
    """
    txt_files = sorted(ocr_dir.glob("*.txt"))
    if not txt_files:
        return "", start_page
    
    parts = []
    current_page = start_page
    
    for txt_file in txt_files:
        # ページ番号を抽出
        match = re.search(r'page_(\d+)\.txt', txt_file.name)
        page_num = int(match.group(1)) if match else current_page
        
        # ページ区切りを挿入
        parts.append(f"\n--- Page {page_num} ---\n")
        
        # テキストを読み込んで後処理
        text = txt_file.read_text(encoding="utf-8")
        text = _post_process_text(text)
        parts.append(text)
        
        current_page = page_num + 1
    
    return "\n".join(parts), current_page


def _ocr_batch(batch_dir: Path, output_dir: Path, batch_num: int, total_batches: int) -> str:
    """
    1バッチ（複数ページ）をOCR処理する。
    
    Returns:
        結合されたテキスト
    """
    logger.info(f"  バッチ {batch_num}/{total_batches} 処理中...")
    
    result = subprocess.run(
        [NDLOCR_CMD, "--sourcedir", str(batch_dir), "--output", str(output_dir)],
        capture_output=True,
        text=True,
        timeout=OCR_TIMEOUT_SEC,
    )
    
    if result.returncode != 0:
        logger.error(f"  バッチ {batch_num} でエラー: {result.stderr}")
        return ""
    
    return _collect_ocr_texts(output_dir)


def ocr_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int = 200,
    skip_embedded: bool = True,
    batch_size: int = BATCH_SIZE,
) -> Path | None:
    """
    PDFにOCRを実行し、テキストファイルを出力する。

    Args:
        pdf_path: 入力PDFのパス
        output_dir: テキスト出力先ディレクトリ
        dpi: PDF→画像変換時の解像度
        skip_embedded: テキスト埋め込み済みPDFをスキップするか
        batch_size: 1バッチあたりのページ数

    Returns:
        結合済みテキストファイルのパス。スキップ時は None。
    """
    pdf_file = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # テキスト埋め込みチェック
    if skip_embedded and has_embedded_text(str(pdf_file)):
        logger.info(f"[SKIP] テキスト埋め込み済み: {pdf_file.name}")
        import fitz
        doc = fitz.open(str(pdf_file))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        out_path = out_dir / f"{pdf_file.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        logger.info(f"[SKIP] テキスト抽出完了: {out_path}")
        return out_path

    # Step 1: PDF → PNG（全ページ変換）
    logger.info(f"[1/3] PDF変換中: {pdf_file.name}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pages_dir = tmpdir / "pages"
        all_images = pdf_to_images(str(pdf_file), str(pages_dir), dpi=dpi)
        total_pages = len(all_images)
        logger.info(f"       {total_pages} ページ変換完了")

        # Step 2: バッチ処理でOCR
        logger.info(f"[2/3] OCR実行中（バッチサイズ: {batch_size}ページ）...")
        all_texts = []
        total_batches = (total_pages + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_pages)
            batch_images = all_images[start_idx:end_idx]
            
            # バッチ用の一時ディレクトリ作成
            batch_tmpdir = Path(tempfile.mkdtemp())
            batch_input = batch_tmpdir / "input"
            batch_output = batch_tmpdir / "output"
            batch_input.mkdir()
            batch_output.mkdir()
            
            # バッチの画像をコピー
            for img_path in batch_images:
                shutil.copy(img_path, batch_input / img_path.name)
            
            # OCR実行
            try:
                batch_text = _ocr_batch(batch_input, batch_output, batch_num + 1, total_batches)
                all_texts.append(batch_text)
                logger.info(f"  バッチ {batch_num + 1}/{total_batches} 完了 ({len(batch_images)}ページ)")
            except subprocess.TimeoutExpired:
                logger.error(f"  バッチ {batch_num + 1}/{total_batches} タイムアウト")
                all_texts.append(f"\n[バッチ {batch_num + 1} タイムアウト]\n")
            except Exception as e:
                logger.error(f"  バッチ {batch_num + 1}/{total_batches} エラー: {e}")
                all_texts.append(f"\n[バッチ {batch_num + 1} エラー: {e}]\n")
            finally:
                # 一時ディレクトリ削除
                shutil.rmtree(batch_tmpdir, ignore_errors=True)

        # Step 3: テキスト結合
        logger.info(f"[3/3] テキスト結合中...")
        combined_text = "\n".join(all_texts)

        if not combined_text.strip():
            logger.warning(f"OCR結果が空です: {pdf_file.name}（DPIを上げて再試行を推奨）")

        out_path = out_dir / f"{pdf_file.stem}.txt"
        out_path.write_text(combined_text, encoding="utf-8")
        logger.info(f"完了: {out_path}")

    return out_path


def batch_ocr(
    input_dir: str,
    output_dir: str,
    dpi: int = 200,
    skip_embedded: bool = True,
    workers: int = 1,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    ディレクトリ内の全PDFを一括処理する。

    Args:
        input_dir: PDFが入っているディレクトリ
        output_dir: テキスト出力先ディレクトリ
        dpi: 解像度
        skip_embedded: テキスト埋め込み済みPDFをスキップするか
        workers: 並列処理ワーカー数（1 = 逐次処理）
        batch_size: 1バッチあたりのページ数
    """
    _check_ndlocr()

    input_path = Path(input_dir)
    pdf_files = sorted(input_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"PDFが見つかりません: {input_path}")
        return

    logger.info(f"{len(pdf_files)} 件のPDFを処理します（workers={workers}）")

    if workers <= 1:
        # 逐次処理
        for i, pdf in enumerate(pdf_files, 1):
            logger.info(f"--- [{i}/{len(pdf_files)}] {pdf.name} ---")
            try:
                ocr_pdf(str(pdf), output_dir, dpi=dpi, skip_embedded=skip_embedded, batch_size=batch_size)
            except Exception as e:
                logger.error(f"ERROR [{pdf.name}]: {e}")
    else:
        # 並列処理
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    ocr_pdf, str(pdf), output_dir, dpi, skip_embedded, batch_size
                ): pdf
                for pdf in pdf_files
            }
            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"ERROR [{pdf.name}]: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF → NDLOCR-Lite OCR pipeline")
    parser.add_argument("input", help="PDFファイルまたはPDFが入ったディレクトリ")
    parser.add_argument("output", help="テキスト出力先ディレクトリ")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"解像度（デフォルト: {DEFAULT_DPI}）")
    parser.add_argument(
        "--no-skip-embedded",
        action="store_true",
        help="テキスト埋め込み済みPDFもOCRする",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列処理ワーカー数（デフォルト: 1 = 逐次）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"1バッチあたりのページ数（デフォルト: {BATCH_SIZE}）",
    )
    args = parser.parse_args()

    _check_ndlocr()

    input_path = Path(args.input)
    skip = not args.no_skip_embedded

    if input_path.is_dir():
        batch_ocr(
            str(input_path), args.output,
            dpi=args.dpi, skip_embedded=skip, workers=args.workers, batch_size=args.batch_size,
        )
    elif input_path.suffix.lower() == ".pdf":
        ocr_pdf(str(input_path), args.output, dpi=args.dpi, skip_embedded=skip, batch_size=args.batch_size)
    else:
        logger.error("入力はPDFファイルまたはディレクトリを指定してください")
        sys.exit(1)
```

---

## 使い方

### 単一PDFの処理

```bash
python pdf_to_text.py input/document.pdf output/
```

### ディレクトリ内のPDFを一括処理

```bash
python pdf_to_text.py input/ output/
```

### 解像度を上げて精度向上（処理は遅くなる）

```bash
python pdf_to_text.py input/document.pdf output/ --dpi 300
```

### テキスト埋め込みPDFもOCRする（通常はスキップされる）

```bash
python pdf_to_text.py input/ output/ --no-skip-embedded
```

### 並列処理で高速化

```bash
python pdf_to_text.py input/ output/ --workers 4
```

### バッチサイズの調整

ページ数が多いPDFはバッチ分割して処理する。小さいバッチはタイムアウトしにくいが全体処理時間は増える。

```bash
# 小さいバッチ（10ページ/バッチ）- タイムアウト回避向け
python pdf_to_text.py input/document.pdf output/ --batch-size 10

# 大きいバッチ（50ページ/バッチ）- 高速化向け
python pdf_to_text.py input/document.pdf output/ --batch-size 50
```

### 出力ファイル

`output/` 以下にPDFと同名の `.txt` ファイルが生成される。

```
output/
├── document.txt
└── report.txt
```

---

## パラメータ指針

| DPI | 用途 | 速度 |
|-----|------|------|
| 150 | 大量処理・速度優先 | 速い |
| 200 | 速度優先 | 普通 |
| 300 | 通常用途（デフォルト）・精度向上 | やや遅い |
| 400 | 高精度・細かい文字 | 遅い |

| workers | 用途 | 備考 |
|---------|------|------|
| 1 | デフォルト（逐次処理） | メモリ消費最小 |
| 2–4 | バッチ処理の高速化 | CPU コア数に応じて調整 |

| batch_size | 用途 | 備考 |
|------------|------|------|
| 10 | タイムアウト回避 | 大きいPDF・スキャン画質が悪い場合向け |
| 20 | 通常（デフォルト） | バランスの取れた設定 |
| 50 | 高速化 | 小さいPDF・高スペック環境向け |

---

## 注意事項

- **対応PDF**: スキャン画像型PDF。テキスト埋め込み済みPDFは自動検出してテキスト抽出にフォールバックする（`--no-skip-embedded` で強制OCRも可）。
- **古典籍・くずし字**: 本パイプラインは現代活字向け。くずし字は `ndl-lab/ndlkotenocr-lite` を使うこと。
- **処理時間**: ノートPCのCPUで1ページ数秒〜十数秒程度（モデルのロード時間が初回にかかる）。
- **一時ファイル**: `tempfile.TemporaryDirectory()` により処理後に自動削除される。
- **CLI互換性**: ndlocr-lite のバージョンによりCLI引数や出力形式が変わる可能性がある。導入時に `ndlocr-lite --help` で確認すること。

---

## 依存関係

```
# requirements.txt
pymupdf>=1.24.0
```

`ndlocr-lite` 本体は `uv tool install .` でグローバルにインストール済みであること。

---

## トラブルシューティング

**`ndlocr-lite: command not found`**
→ `uv tool install .` を ndlocr-lite リポジトリ内で実行する。スクリプト実行時にも起動前チェックでエラーメッセージが表示される。

**OCR結果が空**
→ DPIを上げる（`--dpi 300`）。ログに「OCR出力に .txt が見つかりません」と出る場合は ndlocr-lite の出力形式（XML等）を確認し `_collect_ocr_texts()` を修正する。

**テキスト埋め込み済みPDFが誤検出される**
→ `--no-skip-embedded` で強制OCR。または `has_embedded_text()` の `threshold` を調整する。

**メモリ不足**
→ `--workers 1`（デフォルト）で逐次処理にする。大量ページのPDFはバッチサイズを分割して処理する。

**タイムアウト**
→ バッチサイズを小さくする（`--batch-size 10`）。または `OCR_TIMEOUT_SEC`（デフォルト300秒/バッチ）を増やす。

**OCRが途中で止まる**
→ ndlocr-lite はページ数が多いと処理後に終了しないことがある。バッチ処理（デフォルト20ページ/バッチ）で回避可能。
