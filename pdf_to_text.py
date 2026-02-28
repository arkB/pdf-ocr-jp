import subprocess
import shutil
import sys
import tempfile
import logging
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdf_to_pages import pdf_to_images, has_embedded_text

# Janomeのインポート（オプション）
JANOME_AVAILABLE = False
_janome_tokenizer = None

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer

    JANOME_AVAILABLE = True
except ImportError:
    pass

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
DEFAULT_MAX_LINE_LENGTH = 70  # 1行の最大文字数（デフォルト70文字）


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
        "Iinux": "Linux",
        "Iinuxを": "Linuxを",
        "Iinuxは": "Linuxは",
        "Iinuxの": "Linuxの",
        "Iinuxが": "Linuxが",
        "Iinuxに": "Linuxに",
        "Iinuxで": "Linuxで",
        "Iinuxと": "Linuxと",
        # 人名の誤認識
        "武内党": "武内覚",
        "党訳": "覚訳",
        # カタカナの誤認識
        "ウェプ": "ウェブ",
        "ペース": "ベース",
        # 漢字の誤認識
        "活��": "活用",
        "興味": "興味",  # 興は旧字体だが、興味は正しい
        "快諾": "快諾",  # これも正しい表記
        # 記号の正規化
        "TM、、": "™、®、©",
        "TM,": "™,",
        "(R)": "®",
        "(C)": "©",
        "(TM)": "™",
        # 数字の修正
        "202年": "2022年",
        "201年": "2021年",
        "200年": "2000年",
        # スペースの正規化
        "  ": " ",
        "　　": "　",
    }

    # 誤認識の修正（単語境界を考慮）
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    # 連続する空白を1つに
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"　{2,}", "　", text)

    # 行末の空白を削除
    text = re.sub(r" +\n", "\n", text)

    # 日本語の改行を整形（文節単位で結合）
    text = _fix_japanese_line_breaks(text)

    return text


def _segment_japanese_line(
    line: str, max_chars: int = DEFAULT_MAX_LINE_LENGTH
) -> list[str]:
    """
    Janomeを使って日本語の文章を文節単位に分割する。
    読点（、）は文節の区切りとして扱い、句点（。）で段落を区切る。
    英語が含まれる場合は単語単位で改行する。
    固有名詞やカタカナ語は分割しない。
    max_charsを目安に、前後の範囲内で最適な改行位置を探す。

    Args:
        line: 分割する文章
        max_chars: 1行の目標文字数（目安）

    Returns:
        分割された行のリスト
    """
    global _janome_tokenizer

    # 許容範囲内の文字数ならそのまま返す
    min_chars = int(max_chars * 0.75)  # 最小75%
    max_limit = int(max_chars * 1.2)  # 最大120%

    if len(line) <= max_limit:
        return [line]

    # 英語のみの行は単語単位で改行
    if all(c.isascii() or c.isspace() or c in ".,!?-:;/" for c in line):
        return _segment_english_line(line, max_chars)

    # 日本語が含まれていない、またはJanomeが利用不可ならそのまま返す
    if not JANOME_AVAILABLE:
        return [line]

    # Janomeトークナイザーを初期化（初回のみ）
    if _janome_tokenizer is None:
        _janome_tokenizer = JanomeTokenizer()

    tokenizer = _janome_tokenizer
    tokens = list(tokenizer.tokenize(line))

    # 固有名詞とカタカナ語を結合
    tokens = _merge_proper_nouns(tokens)

    result = []
    current_line = ""
    current_length = 0

    for i, token in enumerate(tokens):
        word = token.surface
        pos = token.part_of_speech.split(",")[0]
        word_len = len(word)

        # 句点（。）の場合は現在の行を確定して新しい段落を開始
        if word == "。":
            current_line += word
            if current_line.strip():
                result.append(current_line)
            current_line = ""
            current_length = 0
            continue

        # 最大限界を超えた場合は強制改行
        if current_length + word_len > max_limit and current_line:
            result.append(current_line)
            current_line = word
            current_length = word_len
            continue

        # 現在の行に追加
        current_line += word
        current_length += word_len

        # 目標文字数を超えた場合で、良い改行位置（読点で終わる）なら改行
        if current_length >= max_chars and current_line.endswith("、"):
            result.append(current_line)
            current_line = ""
            current_length = 0

    # 残りを追加
    if current_line.strip():
        result.append(current_line)

    return result if result else [line]


class _SimpleToken:
    """結合したトークンを表す簡易クラス"""

    def __init__(self, surface, pos):
        self.surface = surface
        self.part_of_speech = pos


def _merge_proper_nouns(tokens):
    """
    固有名詞とカタカナ語を結合する。
    例: "Linux" + "エンジニア" → "Linuxエンジニア"
         "プログ" + "ラミング" → "プログラミング"
         "い" + "た" → "いた"
    """
    if not tokens:
        return tokens

    # 結合すべき固有名詞のパターン
    proper_noun_patterns = [
        ("Linux", "エンジニア"),
        ("Linux", "カーネル"),
        ("Linux", "ディストリビューション"),
        ("Linux", "システム"),
        ("プログ", "ラミング"),
        ("子", "プロセス"),
        ("コンテナ", "技術"),
        ("オペレーティング", "システム"),
        ("オープン", "ソース"),
        ("コマンド", "ライン"),
        ("ファイル", "システム"),
        ("ネットワーク", "インターフェース"),
        ("クラウド", "ネイティブ"),
        ("システム", "コール"),
    ]

    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        word = token.surface
        pos = token.part_of_speech.split(",")[0]

        # 次のトークンと結合すべきかチェック
        merged = False
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]
            next_word = next_token.surface

            # パターンに一致するかチェック
            for pattern1, pattern2 in proper_noun_patterns:
                if word == pattern1 and next_word == pattern2:
                    # 結合した新しいトークンを作成
                    merged_word = word + next_word

                    merged_token = _SimpleToken(merged_word, "名詞,固有名詞")
                    result.append(merged_token)
                    i += 2
                    merged = True
                    break

            # カタカナ語の連続を結合
            if not merged and _is_katakana(word) and _is_katakana(next_word):
                # カタカナ語を結合
                merged_word = word + next_word

                merged_token = _SimpleToken(merged_word, "名詞,一般")
                # さらに次のカタカナ語も結合
                j = i + 2
                while j < len(tokens) and _is_katakana(tokens[j].surface):
                    merged_word += tokens[j].surface
                    merged_token.surface = merged_word
                    j += 1

                result.append(merged_token)
                i = j
                merged = True

        # 助動詞の連続を結合（例: い + た → いた）
        if not merged and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            # 「い」+「た」、「て」+「いる」などの助動詞連続
            if pos in ["助動詞", "動詞"] and word in ["い", "て", "き", "し"]:
                next_pos = next_token.part_of_speech.split(",")[0]
                if next_pos in ["助動詞", "動詞", "形容詞"]:
                    merged_word = word + next_token.surface
                    merged_token = _SimpleToken(merged_word, "助動詞")
                    result.append(merged_token)
                    i += 2
                    merged = True

        if not merged:
            result.append(token)
            i += 1

    return result


def _is_katakana(text):
    """
    テキストがカタカナのみで構成されているかチェック
    """
    if not text:
        return False
    return all("\u30a0" <= c <= "\u30ff" or c in "ー・" for c in text)


def _segment_english_line(
    line: str, max_chars: int = DEFAULT_MAX_LINE_LENGTH
) -> list[str]:
    """
    英語の文章を単語単位に分割する。

    Args:
        line: 分割する文章
        max_chars: 1行の最大文字数

    Returns:
        分割された行のリスト
    """
    if len(line) <= max_chars:
        return [line]

    import re

    # 単語に分割（スペースで区切る）
    words = re.findall(r"\S+\s*", line)

    result = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) > max_chars and current_line:
            result.append(current_line.rstrip())
            current_line = word
        else:
            current_line += word

    if current_line:
        result.append(current_line.rstrip())

    return result if result else [line]


def _fix_japanese_line_breaks(text: str) -> str:
    """
    日本語テキストの改行を整形する。
    文節の途中での改行を削除し、句読点で区切られた段落構造を維持する。
    """
    lines = text.split("\n")
    result = []
    i = 0

    # 助詞のリスト（行末に来た場合、次の行と結合すべき）
    particles = [
        "て",
        "で",
        "に",
        "を",
        "は",
        "が",
        "の",
        "と",
        "から",
        "より",
        "も",
        "や",
        "か",
        "ば",
        "と",
        "へ",
        "まで",
        "だけ",
        "しか",
        "など",
        "ながら",
        "つつ",
        "ばかり",
        "ほど",
        "くらい",
        "くらい",
        "など",
        "という",
        "とか",
        "および",
        "または",
        "ならびに",
        "また",
    ]

    # 行頭に来た場合、前の行と結合すべき文字
    continue_chars_start = [
        "て",
        "で",
        "に",
        "を",
        "は",
        "が",
        "の",
        "と",
        "も",
        "や",
        "から",
        "より",
        "など",
        "しか",
        "だけ",
        "ばかり",
        "ほど",
        "くらい",
        "ぐらい",
        "など",
        "などの",
        "ある",
        "いる",
        "する",
        "なり",
        "ます",
        "です",
        "した",
        "したが",
        "して",
        "され",
        "されて",
        "られる",
        "られて",
        "こと",
        "もの",
        "ため",
        "ように",
        "ような",
        "として",
        "による",
        "について",
        "に対して",
        "によって",
        "において",
        "に関して",
        "については",
        "「",
        "（",
        "『",
        "【",
        "［",
        "〈",
        "《",
        "≪",
        "＜",
    ]

    # 行末に来た場合、次の行と結合すべき文字（助詞や接続詞など）
    continue_chars_end = [
        "、",
        "・",
        "…",
        "...",
        "．",
        "。",
        "！",
        "？",
        "」",
        "）",
        "』",
        "】",
        "］",
        "〉",
        "》",
        "≫",
        "＞",
        "：",
        "；",
    ]

    # 固有名詞の接尾辞（これで終わる行は次と結合すべき）
    proper_noun_suffixes = [
        "Linuxエンジ",
        "プログ",
        "子プロ",
        "システムコ",
        "コンテナ",
        "オペレーティングシステ",
        "ディストリビュー",
        "インフラストラク",
        "アーキテク",
        "ドキュメ",
        "インテ",
        "アプ",
        "サーバ",
        "ネットワーク",
        "データベ",
        "プラットフォ",
        "ライブラ",
        "フレームワーク",
        "マネージ",
        "モニタ",
        "デバッ",
    ]

    # 固有名詞の接頭辞（これで始まる行は前と結合すべき）
    proper_noun_prefixes = [
        "ニア",
        "ラミング",
        "セス",
        "ック",
        "ンテナ",
        "ム",
        "ション",
        "チャ",
        "ックチャ",
        "リケーション",
        "ール",
        "ワーク",
        "ス",
        "ータ",
        "イザ",
    ]

    # 行末に来た場合、次の行と結合すべき助詞・接続詞など
    particles_end = [
        "て",
        "で",
        "に",
        "を",
        "は",
        "が",
        "の",
        "と",
        "から",
        "より",
        "も",
        "や",
        "か",
        "ば",
        "へ",
        "まで",
        "だけ",
        "しか",
        "など",
        "ながら",
        "つつ",
        "ばかり",
        "ほど",
        "くらい",
        "ぐらい",
        "という",
        "とか",
        "または",
        "ならびに",
        "また",
    ]

    # 行頭に来た場合、前の行と結合すべき助詞・活用語尾など
    particles_start = [
        "て",
        "で",
        "に",
        "を",
        "は",
        "が",
        "の",
        "と",
        "も",
        "や",
        "から",
        "より",
        "など",
        "しか",
        "だけ",
        "ばかり",
        "ほど",
        "くらい",
        "ぐらい",
        "ある",
        "いる",
        "する",
        "なり",
        "ます",
        "です",
        "した",
        "して",
        "され",
        "られ",
        "こと",
        "もの",
    ]

    # 行をまたいで結合すべきでない記号（段落の区切りとして保持）
    sentence_end_marks = [
        "。",
        "！",
        "？",
        "」",
        "）",
        "』",
        "】",
        "…",
        "...",
        "!",
        "?",
    ]

    while i < len(lines):
        line = lines[i]

        # ページ区切りや空行はそのまま保持
        if line.strip().startswith("--- Page") or line.strip() == "":
            result.append(line)
            i += 1
            continue

        current_line = line.rstrip()

        # 次の行と結合すべきか判断
        while i + 1 < len(lines):
            next_line = lines[i + 1]

            # 次の行がページ区切りや空行なら結合しない
            if next_line.strip().startswith("--- Page") or next_line.strip() == "":
                break

            next_content = next_line.lstrip()
            if not next_content:
                break

            should_merge = False

            current_stripped = current_line.lstrip()
            current_len = len(current_stripped)

            # 短すぎる行（6文字未満）は表題などとして段落として扱う
            if current_len >= 6:
                # 結合条件: 現在の行が句読点で終わっていない
                if not any(current_line.endswith(mark) for mark in sentence_end_marks):
                    # 条件1: 行末が助詞で終わる（文節の途中で改行されている）
                    if any(current_line.endswith(p) for p in particles_end):
                        should_merge = True
                    # 条件2: 行末が固有名詞の途中（例: Linuxエンジ）
                    elif any(
                        current_line.endswith(suffix) for suffix in proper_noun_suffixes
                    ):
                        should_merge = True
                    # 条件3: 次の行が固有名詞の接頭辞で始まる（例: ニア）
                    elif any(
                        next_content.startswith(prefix)
                        for prefix in proper_noun_prefixes
                    ):
                        should_merge = True
                    # 条件4: 行末がひらがな1文字で終わる（文節の途中）
                    elif (
                        current_line
                        and "\u3040" <= current_line[-1] <= "\u309f"
                        and len(current_line[-1]) == 1
                    ):
                        # ただし「です」「ます」や名詞（まえがき、ついて など）は除く
                        noun_endings = [
                            "です",
                            "ます",
                            "した",
                            "しい",
                            "うち",
                            "おり",
                            "がき",
                            "して",
                            "って",
                            "いて",
                            "について",
                        ]
                        if not any(current_line.endswith(end) for end in noun_endings):
                            should_merge = True
                    # 条件5: 現在の行が短い（15文字未満）かつ次の行も短い→結合
                    elif current_len < 15 and len(next_content) < 30:
                        should_merge = True
                    # 条件6: 次の行が助詞で始まる（文節の途中）
                    elif any(next_content.startswith(p) for p in particles_start):
                        if current_len >= 12:
                            should_merge = True
                    # 条件7: 行末が「、」で終わる
                    elif current_line.endswith("。"):
                        if current_len >= 12:
                            should_merge = True
                    # 条件8: 現在の行が名詞っぽく、次の行が活用語尾や助詞で始まる（名詞の途中）
                    elif (
                        current_len < 80
                        and next_content
                        and (
                            # 次がひらがな1-3文字で始まり、短い行（活用語尾の可能性）
                            (
                                all(
                                    "\u3040" <= c <= "\u309f"
                                    for c in next_content[: min(3, len(next_content))]
                                )
                                and len(next_content) <= 10
                            )
                            or
                            # 次が「れ」「て」「に」などの助詞や接続詞で始まる
                            any(
                                next_content.startswith(p)
                                for p in [
                                    "れ",
                                    "て",
                                    "に",
                                    "で",
                                    "が",
                                    "を",
                                    "は",
                                    "の",
                                    "と",
                                    "も",
                                    "や",
                                ]
                            )
                        )
                    ):
                        should_merge = True
                    elif current_line.endswith("、"):
                        if current_len >= 15:
                            should_merge = True

            if should_merge:
                # 行を結合（スペースなしで結合）
                current_line = current_line + next_content
                i += 1
            else:
                break

        result.append(current_line)
        i += 1

    return "\n".join(result)


def _collect_ocr_texts(ocr_dir: Path, start_page: int = 1) -> tuple[str, int]:
    """
    ndlocr-lite の出力ディレクトリからテキストを収集・結合する。
    ページ区切りを挿入し、後処理を適用する。

    Args:
        ocr_dir: OCR出力ディレクトリ
        start_page: 開始ページ番号

    Returns:
        (結合されたテキスト, 最終ページ番号)
    """
    txt_files = sorted(ocr_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(
            "OCR出力に .txt が見つかりません。"
            "ndlocr-lite の出力形式を確認してください（XML等の可能性あり）。"
        )
        return "", start_page

    parts = []
    current_page = start_page

    for txt_file in txt_files:
        # ページ番号を抽出（page_XXXX.txtから）
        match = re.search(r"page_(\d+)\.txt", txt_file.name)
        if match:
            page_num = int(match.group(1))
        else:
            page_num = current_page

        # ページ区切りを挿入
        parts.append(f"\n--- Page {page_num} ---\n")

        # テキストを読み込んで後処理
        text = txt_file.read_text(encoding="utf-8")
        text = _post_process_text(text)
        parts.append(text)

        current_page = page_num + 1

    return "\n".join(parts), current_page


def _ocr_batch(
    batch_dir: Path,
    output_dir: Path,
    batch_num: int,
    total_batches: int,
    start_page: int = 1,
) -> tuple[str, int]:
    """
    1バッチ（複数ページ）をOCR処理する。

    Args:
        batch_dir: 入力画像ディレクトリ
        output_dir: OCR出力ディレクトリ
        batch_num: 現在のバッチ番号
        total_batches: 総バッチ数
        start_page: 開始ページ番号

    Returns:
        (結合されたテキスト, 最終ページ番号)
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
        return "", start_page

    return _collect_ocr_texts(output_dir, start_page)


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
        logger.info(
            f"[2/3] OCR実行中（バッチサイズ: {batch_size}ページ, DPI: {dpi}）..."
        )
        all_texts = []
        total_batches = (total_pages + batch_size - 1) // batch_size
        current_page = 1  # ページ番号追跡

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

            # バッチの画像をコピー（ページ番号を維持）
            for img_path in batch_images:
                shutil.copy(img_path, batch_input / img_path.name)

            # OCR実行
            try:
                batch_text, last_page = _ocr_batch(
                    batch_input,
                    batch_output,
                    batch_num + 1,
                    total_batches,
                    current_page,
                )
                all_texts.append(batch_text)
                current_page = last_page  # 次のバッチの開始ページを更新
                logger.info(
                    f"  バッチ {batch_num + 1}/{total_batches} 完了 ({len(batch_images)}ページ)"
                )
            except subprocess.TimeoutExpired:
                logger.error(f"  バッチ {batch_num + 1}/{total_batches} タイムアウト")
                all_texts.append(
                    f"\n--- Page {current_page} ---\n[バッチ {batch_num + 1} タイムアウト]\n"
                )
                current_page += len(batch_images)
            except Exception as e:
                logger.error(f"  バッチ {batch_num + 1}/{total_batches} エラー: {e}")
                all_texts.append(
                    f"\n--- Page {current_page} ---\n[バッチ {batch_num + 1} エラー: {e}]\n"
                )
                current_page += len(batch_images)
            finally:
                # 一時ディレクトリ削除
                shutil.rmtree(batch_tmpdir, ignore_errors=True)

        # Step 3: テキスト結合
        logger.info(f"[3/3] テキスト結合中...")
        combined_text = "\n".join(all_texts)

        if not combined_text.strip():
            logger.warning(
                f"OCR結果が空です: {pdf_file.name}（DPIを上げて再試行を推奨）"
            )

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
                ocr_pdf(
                    str(pdf),
                    output_dir,
                    dpi=dpi,
                    skip_embedded=skip_embedded,
                    batch_size=batch_size,
                )
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
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"解像度（デフォルト: {DEFAULT_DPI}）",
    )
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
            str(input_path),
            args.output,
            dpi=args.dpi,
            skip_embedded=skip,
            workers=args.workers,
            batch_size=args.batch_size,
        )
    elif input_path.suffix.lower() == ".pdf":
        ocr_pdf(
            str(input_path),
            args.output,
            dpi=args.dpi,
            skip_embedded=skip,
            batch_size=args.batch_size,
        )
    else:
        logger.error("入力はPDFファイルまたはディレクトリを指定してください")
        sys.exit(1)
