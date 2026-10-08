"""OCR engines for scanned pages and image regions.

Both engines return the same OcrResult: text blocks and tables with
positions in PDF points (the coordinates PyMuPDF uses), so the parser
can merge OCR output with text-layer output and everything downstream
(chunking, coverage checks, citations) works unchanged.

  easyocr  Pillow-preprocessed page image -> EasyOCR text lines, laid
           out here into paragraphs and (column-aligned) tables.
  docling  Docling's layout and TableFormer models (with EasyOCR inside)
           on the page itself: paragraphs and real table structure.

Models load on first use (and download once: EasyOCR ~100 MB to
~/.EasyOCR, Docling's layout/table models to the Hugging Face cache).
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from statistics import median
import warnings

import pymupdf

from .. import config
from ..chunking.table_chunker import NUMBER, TABLE_NOTE, TABLE_TITLE, YEAR

# "1. City service overview": a numbered heading, kept as its own block
# like a PDF text layer does, so chunking sees section boundaries.
SECTION_HEADING = re.compile(r"^\d+\.\s+\S.{0,80}$")

# EasyOCR often reads "$" as "S". Fixed only directly before an amount
# ("S185,000", "S6.3M", "S1.2bn"), never in words or codes like "S3".
_DOLLAR_AS_S = re.compile(
    r"(?<![\w$])S(?=\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\d+(?:\.\d+)?(?:[MKB]|bn|m|k)\b)")

# Text regions whose OCR segments average fewer characters than this,
# with no table found, are figures (chart labels, axis ticks, legends).
FIGURE_MAX_MEAN_CHARS = 8

# Harmless deprecation/MPS notices torch prints from inside EasyOCR.
_TORCH_NOISE = r".*(quantize_per_tensor|quantized tensor|pin_memory).*"


def fix_currency(text: str) -> str:
    return _DOLLAR_AS_S.sub("$", text)


class OcrUnavailable(RuntimeError):
    """The engine's package is missing; the message says how to install."""


@dataclass
class OcrResult:
    # {"bbox": [x0, y0, x1, y1], "text": str, "confidence": float | None}
    blocks: list[dict] = field(default_factory=list)
    # {"bbox": [x0, y0, x1, y1], "rows": [[cell, ...], ...],
    #  "confidence": float | None}
    tables: list[dict] = field(default_factory=list)
    engine: str = ""
    dropped_lines: int = 0

    @property
    def mean_confidence(self) -> float | None:
        scores = [item["confidence"] for item in self.blocks + self.tables
                  if item.get("confidence") is not None]
        return sum(scores) / len(scores) if scores else None


# --- image preparation (Pillow) ------------------------------------------

def render_page(page: pymupdf.Page, dpi: int, clip=None):
    """The page (or the `clip` rectangle of it) as a grayscale image."""
    from PIL import Image

    pixmap = page.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csGRAY,
                             alpha=False)
    return Image.frombytes("L", (pixmap.width, pixmap.height),
                           pixmap.samples)


def prepare_image(image):
    """Stretch contrast so faint or greyish scans read like clean print."""
    from PIL import ImageOps

    return ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1)


# --- grouping OCR lines into paragraphs -------------------------------------

def _box(points) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _union(boxes) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _height(box) -> float:
    return box[3] - box[1]


def _lines_and_pieces(segments: list[dict]) -> list[list[list[dict]]]:
    """Segments -> text lines (top to bottom), each split into pieces.

    Segments with a similar vertical centre form one line. Inside a
    line, a horizontal gap wider than ~2.5 character heights starts a
    new piece: a table column or a second text column, not a word gap.
    """
    rows: list[list[dict]] = []
    for segment in sorted(segments,
                          key=lambda s: ((s["bbox"][1] + s["bbox"][3]) / 2,
                                         s["bbox"][0])):
        centre = (segment["bbox"][1] + segment["bbox"][3]) / 2
        if rows:
            row_box = _union([s["bbox"] for s in rows[-1]])
            row_centre = (row_box[1] + row_box[3]) / 2
            if abs(centre - row_centre) <= 0.5 * max(
                    _height(segment["bbox"]), _height(row_box)):
                rows[-1].append(segment)
                continue
        rows.append([segment])

    lines = []
    for row in rows:
        row.sort(key=lambda s: s["bbox"][0])
        pieces = [[row[0]]]
        for segment in row[1:]:
            gap = segment["bbox"][0] - pieces[-1][-1]["bbox"][2]
            if gap > 2.5 * max(_height(segment["bbox"]),
                               _height(pieces[-1][-1]["bbox"])):
                pieces.append([segment])
            else:
                pieces[-1].append(segment)
        lines.append(pieces)
    return lines


def _piece(segments: list[dict]) -> dict:
    return {"bbox": _union([s["bbox"] for s in segments]),
            "text": " ".join(s["text"].strip() for s in segments),
            "segments": segments}


def _find_tables(lines: list[list[dict]]) -> list[tuple[int, int, list[float]]]:
    """Runs of lines whose pieces line up in columns: (first, last, columns).

    A run needs 2+ lines with 2+ pieces each, the pieces starting at
    shared column positions, cell-like pieces (narrow compared with the
    run), and a header-like first line: every piece text, not numbers
    (years allowed). Two columns of prose have wide pieces, and chart
    axis labels are numbers, so neither becomes a table.
    """
    runs = []
    index = 0
    while index < len(lines):
        if len(lines[index]) < 2:
            index += 1
            continue
        height = median(_height(p["bbox"]) for p in lines[index])
        tolerance = 1.5 * height
        columns = [p["bbox"][0] for p in lines[index]]
        end = index
        while end + 1 < len(lines):
            following = lines[end + 1]
            gap = following[0]["bbox"][1] - max(p["bbox"][3] for p in lines[end])
            matched = sum(any(abs(p["bbox"][0] - c) <= tolerance for c in columns)
                          for p in following)
            if len(following) < 2 or matched < 2 or gap > 2.5 * height:
                break
            for p in following:  # a row may fill a column the first lacked
                if not any(abs(p["bbox"][0] - c) <= tolerance for c in columns):
                    columns.append(p["bbox"][0])
            end += 1

        pieces = [p for line in lines[index:end + 1] for p in line]
        run_box = _union([p["bbox"] for p in pieces])
        widths = [p["bbox"][2] - p["bbox"][0] for p in pieces]
        if (end > index
                and median(widths) <= 0.35 * (run_box[2] - run_box[0])
                and _header_like(lines[index])):
            runs.append((index, end, sorted(columns)))
            index = end + 1
        else:
            index += 1
    return runs


def _header_like(line: list[dict]) -> bool:
    texts = [p["text"].strip() for p in line]
    return all(
        any(ch.isalpha() for ch in text) or YEAR.match(text)
        for text in texts
    ) and not any(NUMBER.match(text) and not YEAR.match(text)
                  for text in texts)


def _table_from(lines: list[list[dict]], columns: list[float]) -> dict:
    """Cells by column. Each OCR segment goes to the column it starts in
    (the rightmost column start at or left of it), so two header words
    that sit close together ("Annual Cost" | "Term") still split."""
    height = median(_height(p["bbox"]) for line in lines for p in line)
    rows = []
    for line in lines:
        cells = [[] for _ in columns]
        for segment in (s for p in line for s in p["segments"]):
            x0 = segment["bbox"][0]
            starts = [i for i, c in enumerate(columns) if c <= x0 + 1.5 * height]
            column = starts[-1] if starts else 0
            cells[column].append(segment["text"].strip())
        rows.append([" ".join(cell) for cell in cells])
    pieces = [p for line in lines for p in line]
    segments = [s for p in pieces for s in p["segments"]]
    return {"bbox": _union([p["bbox"] for p in pieces]), "rows": rows,
            "confidence": _confidence(segments)}


def _confidence(segments: list[dict]) -> float:
    weights = [max(1, len(s["text"])) for s in segments]
    return round(sum(s["confidence"] * w for s, w in zip(segments, weights))
                 / sum(weights), 4)


def layout_lines(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """OCR segments -> (paragraph blocks, tables).

    1. Segments become text lines, split into pieces at wide gaps.
    2. Runs of lines whose pieces line up in columns become tables, so
       a scanned table keeps its rows and columns ("Location: East |
       Participants: 85") instead of turning into loose words.
    A region with no table whose segments are short labels (a chart)
    becomes a single figure block instead.
    3. Remaining pieces close below each other and overlapping
       horizontally become paragraph blocks. Section headings stay
       blocks of their own, and table titles ("Table 2. ...") and notes
       ("Note:", "Source:") always start a new block, because chunking
       finds titles and notes by how a block begins.
    Confidence is the length-weighted mean of the segments involved.
    """
    if not segments:
        return [], []

    lines = [[_piece(p) for p in pieces] for pieces in _lines_and_pieces(segments)]
    tables, in_table = [], set()
    for first, last, columns in _find_tables(lines):
        tables.append(_table_from(lines[first:last + 1], columns))
        in_table.update(range(first, last + 1))

    # A chart or diagram: short labels scattered over the image. Keep
    # them as one searchable block instead of many tiny "paragraphs".
    mean_chars = sum(len(s["text"]) for s in segments) / len(segments)
    if not tables and mean_chars < FIGURE_MAX_MEAN_CHARS:
        return [{
            "bbox": _union([s["bbox"] for s in segments]),
            "text": "\n".join(" ".join(p["text"] for p in line)
                              for line in lines) + "\n",
            "confidence": _confidence(segments),
            "figure": True,
        }], []

    paragraphs: list[list[dict]] = []
    closed: set[int] = set()  # ids of heading blocks: nothing joins them
    rest = [p for i, line in enumerate(lines) if i not in in_table for p in line]
    for piece in sorted(rest, key=lambda p: (p["bbox"][1], p["bbox"][0])):
        box = piece["bbox"]
        text = piece["text"].strip()
        heading = bool(SECTION_HEADING.match(text))
        if heading or TABLE_TITLE.match(text) or TABLE_NOTE.match(text):
            paragraphs.append([piece])
            if heading:
                closed.add(id(paragraphs[-1]))
            continue
        for block in reversed(paragraphs[-6:]):  # recent blocks: other columns
            if id(block) in closed:
                continue
            last = block[-1]["bbox"]
            gap = box[1] - last[3]
            overlap = min(box[2], last[2]) - max(box[0], last[0])
            narrower = min(box[2] - box[0], last[2] - last[0]) or 1
            if (-0.5 * _height(box) <= gap <= 0.8 * max(_height(box), _height(last))
                    and overlap >= 0.3 * narrower):
                block.append(piece)
                break
        else:
            paragraphs.append([piece])

    blocks = [{
        "bbox": _union([p["bbox"] for p in block]),
        "text": "\n".join(p["text"] for p in block) + "\n",
        "confidence": _confidence([s for p in block for s in p["segments"]]),
    } for block in paragraphs]
    return blocks, tables


# --- engines -----------------------------------------------------------------

class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, languages=None, dpi=None, min_line_confidence=None):
        self.languages = languages or config.OCR_LANGUAGES
        self.dpi = dpi or config.OCR_DPI
        self.min_line_confidence = (config.OCR_MIN_LINE_CONFIDENCE
                                    if min_line_confidence is None
                                    else min_line_confidence)
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise OcrUnavailable(
                    'EasyOCR is not installed: pip install -e ".[ocr]"'
                ) from exc
            # CPU by default; DOCINTEL_OCR_GPU=1 uses CUDA or Apple's GPU
            # (MPS), which shares memory with everything else on a Mac.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_TORCH_NOISE)
                self._reader = easyocr.Reader(
                    self.languages, gpu=config.OCR_GPU, verbose=False)
        return self._reader

    def read_image(self, image) -> list[dict]:
        """Raw EasyOCR segments: bbox in image pixels, text, confidence."""
        import numpy as np

        reader = self._get_reader()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_TORCH_NOISE)
            detections = reader.readtext(np.asarray(image), detail=1,
                                         paragraph=False)
        return [{"bbox": _box(points), "text": fix_currency(text),
                 "confidence": float(score)}
                for points, text, score in detections if text.strip()]

    def recognize(self, pdf_path: Path, page: pymupdf.Page,
                  clip=None) -> OcrResult:
        image = prepare_image(render_page(page, self.dpi, clip))
        scale = 72 / self.dpi  # pixels -> PDF points
        x_offset, y_offset = (clip.x0, clip.y0) if clip is not None else (0, 0)

        kept, dropped = [], 0
        for segment in self.read_image(image):
            if segment["confidence"] < self.min_line_confidence:
                dropped += 1
                continue
            x0, y0, x1, y1 = segment["bbox"]
            kept.append({**segment, "bbox": [x0 * scale + x_offset,
                                             y0 * scale + y_offset,
                                             x1 * scale + x_offset,
                                             y1 * scale + y_offset]})
        blocks, tables = layout_lines(kept)
        return OcrResult(blocks, tables, self.name, dropped)


class DoclingEngine:
    name = "docling"

    def __init__(self, languages=None):
        self.languages = languages or config.OCR_LANGUAGES
        self._converters = {}

    def _converter(self, full_page: bool):
        if full_page not in self._converters:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    EasyOcrOptions,
                    OcrMode,
                    PdfPipelineOptions,
                )
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError as exc:
                raise OcrUnavailable(
                    'Docling is not installed: pip install -e ".[docling]"'
                ) from exc

            options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                ocr_options=EasyOcrOptions(
                    lang=self.languages,
                    # Scanned page: OCR everything. Mixed page: Docling
                    # skips areas that already have a text layer.
                    mode=OcrMode.FULL_PAGE if full_page else OcrMode.DEFAULT,
                ),
            )
            self._converters[full_page] = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            })
        return self._converters[full_page]

    def recognize(self, pdf_path: Path, page: pymupdf.Page,
                  clip=None) -> OcrResult:
        number = page.number + 1
        result = self._converter(full_page=clip is None).convert(
            Path(pdf_path), page_range=(number, number))
        return items_to_result(result.document, number, clip)


def items_to_result(document, page_no: int, clip=None) -> OcrResult:
    """Docling items on one page -> OcrResult (top-left coordinates).

    For a clip (an image region of a mixed page) only items centred in
    the region are kept; the rest of the page comes from the text layer.
    """
    from docling_core.types.doc import TableItem, TextItem

    height = document.pages[page_no].size.height
    out = OcrResult(engine="docling")

    for item, _level in document.iterate_items(page_no=page_no):
        prov = next((p for p in getattr(item, "prov", None) or []
                     if p.page_no == page_no), None)
        if prov is None:
            continue
        box = prov.bbox.to_top_left_origin(height)
        bbox = [box.l, box.t, box.r, box.b]
        if clip is not None:
            centre = pymupdf.Point((bbox[0] + bbox[2]) / 2,
                                   (bbox[1] + bbox[3]) / 2)
            if not clip.contains(centre):
                continue

        if isinstance(item, TableItem):
            rows = [[cell.text for cell in row] for row in item.data.grid]
            if rows:
                out.tables.append({"bbox": bbox, "rows": rows})
        elif isinstance(item, TextItem) and item.text.strip():
            out.blocks.append({"bbox": bbox, "text": item.text + "\n",
                               "confidence": None})
    return out


ENGINES = {"easyocr": EasyOcrEngine, "docling": DoclingEngine}


def get_ocr_engine(name: str | None = None):
    """The named or configured engine (models load on first use), or None."""
    return _engine((name or config.OCR_ENGINE).lower())


@lru_cache(maxsize=4)
def _engine(name: str):
    # One instance per engine, so models load once per process.
    if name == "none":
        return None
    if name not in ENGINES:
        raise ValueError(f"Unknown OCR engine {name!r}; use one of "
                         f"{sorted(ENGINES)} or 'none'")
    return ENGINES[name]()
