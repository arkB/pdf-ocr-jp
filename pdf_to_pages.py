import fitz  # PyMuPDF
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def has_embedded_text(
    pdf_path: str, sample_pages: int = 3, threshold: int = 50
) -> bool:
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
    pdf_file = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_file))
    image_paths = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72dpiがPDFの基準

    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            img_path = out_dir / f"page_{i + 1:04d}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
            logger.debug(f"  変換完了: {img_path.name}")
    finally:
        doc.close()

    return image_paths
