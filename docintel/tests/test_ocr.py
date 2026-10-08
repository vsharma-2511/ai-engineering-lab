"""OCR layout and engine wiring, without running OCR models.

The layout tests replay real EasyOCR output saved from a scanned copy of
the test PDF (tests/fixtures/ocr_segments_table_page.json).

Run from the project folder:  pytest
"""
import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from docintel import config
from docintel.parsing import ocr
from docintel.parsing.ocr import (
    EasyOcrEngine,
    OcrResult,
    get_ocr_engine,
    layout_lines,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ocr_segments_table_page.json"


@pytest.fixture
def segments():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [s for s in data["segments"]
            if s["confidence"] >= config.OCR_MIN_LINE_CONFIDENCE]


def segment(text, x0, y0, x1, y1, confidence=0.9):
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": confidence}


# --- layout: lines -> paragraphs and tables ---------------------------------

def test_scanned_table_keeps_rows_and_columns(segments):
    blocks, tables = layout_lines(segments)

    assert len(tables) == 1
    assert tables[0]["rows"] == [
        ["Program", "Location", "Participants", "Completion rate"],
        ["Digital skills", "Central", "120", "82%"],
        ["Digital skills", "East", "85", "76%"],
        ["Career workshops", "Central", "64", "91%"],
        ["Career workshops", "East", "48", "88%"],
    ]
    # No table cell is left behind as a loose paragraph of its own.
    cells = {cell for row in tables[0]["rows"] for cell in row}
    assert not any(b["text"].strip() in cells for b in blocks)


def test_paragraph_lines_are_joined_into_blocks(segments):
    blocks, _ = layout_lines(segments)
    texts = [b["text"] for b in blocks]
    heading = texts.index("4. Definitions and caveats\n")  # its own block
    paragraph = blocks[heading + 1]
    assert paragraph["text"].startswith("Completion rate is the percentage")
    assert paragraph["text"].count("\n") == 2   # two lines, one block
    assert 0 < paragraph["confidence"] <= 1


def test_titles_headings_and_notes_start_their_own_blocks():
    """The page with two tables, as EasyOCR read it: chunking needs
    "Table N" titles and "Note:" lines at the start of a block."""
    fixture = Path(__file__).parent / "fixtures" / "ocr_segments_two_tables_page.json"
    segments = [s for s in json.loads(fixture.read_text())["segments"]
                if s["confidence"] >= config.OCR_MIN_LINE_CONFIDENCE]

    blocks, tables = layout_lines(segments)
    firsts = [b["text"].splitlines()[0] for b in blocks]

    assert len(tables) == 2
    for start in ("1. City service overview", "Table 1. Northport library usage",
                  "Note: figures are rounded", "2. Budget summary",
                  "Table 2. Northport library budget (CAD millions)",
                  "Footnote: the 2025 capital amount"):
        assert any(first.startswith(start) for first in firsts), start
    assert "1. City service overview\n" in [b["text"] for b in blocks]


def test_two_columns_of_prose_are_not_a_table():
    lines = []
    for row in range(4):
        y = 100 + row * 14
        lines.append(segment(f"Left column sentence number {row} goes on",
                             50, y, 280, y + 10))
        lines.append(segment(f"Right column sentence number {row} goes on",
                             320, y, 550, y + 10))

    blocks, tables = layout_lines(lines)

    assert tables == []
    assert sorted(b["bbox"][0] for b in blocks) == [50, 320]  # two columns
    assert all(b["text"].count("\n") == 4 for b in blocks)


def test_single_line_with_gaps_is_not_a_table():
    blocks, tables = layout_lines([segment("Name", 50, 100, 80, 110),
                                   segment("Date", 300, 100, 330, 110)])
    assert tables == []
    assert "Name" in blocks[0]["text"] and "Date" in blocks[0]["text"]


def chart_segments():
    """Axis labels of a dual-axis chart, as EasyOCR read the user's
    test document (page 2): numbers in two aligned columns."""
    rows = [("800", "90"), ("700", "88"), ("600", "86"), ("500", "84")]
    out = [segment("Support Volume vs Customer Satisfaction", 204, 372, 414, 386)]
    for i, (left, right) in enumerate(rows):
        y = 397 + i * 30
        out += [segment(left, 138, y, 158, y + 10),
                segment(right, 461, y, 474, y + 10)]
    out.append(segment("Jan Feb Mar Apr", 167, 539, 300, 551))
    return out


def test_chart_axis_numbers_are_not_a_table_but_one_figure_block():
    blocks, tables = layout_lines(chart_segments())
    assert tables == []
    [figure] = blocks
    assert figure["figure"] is True
    assert "Support Volume vs Customer Satisfaction" in figure["text"]
    assert "800" in figure["text"] and "84" in figure["text"]


def test_numeric_first_row_never_starts_a_table():
    rows = [("12", "40"), ("13", "41"), ("14", "42")]
    lines = [segment(text, x, 100 + i * 15, x + 15, 110 + i * 15)
             for i, row in enumerate(rows) for text, x in zip(row, (50, 300))]
    lines.append(segment("A sentence long enough to be prose, not labels.",
                         50, 200, 400, 210))
    assert layout_lines(lines)[1] == []


def test_close_header_words_split_into_their_columns():
    """The user's scanned vendor table: "Annual Cost" and "Term" sit
    closer than a column gap, but the data rows show two columns."""
    fixture = Path(__file__).parent / "fixtures" / "ocr_segments_vendor_table.json"
    segments = [s for s in json.loads(fixture.read_text())["segments"]
                if s["confidence"] >= config.OCR_MIN_LINE_CONFIDENCE]

    _blocks, [table] = layout_lines(segments)

    assert table["rows"] == [
        ["Vendor", "Service", "Annual Cost", "Term"],
        ["Cloudline Inc:", "Hosting", "$1,240,000", "3 years"],
        ["DataHarbor", "Backup", "$185,000", "2 years"],
        ["SecureGate", "Security audit", "$96,500", "1 year"],
    ]


@pytest.mark.parametrize("raw, fixed", [
    ("S185,000", "$185,000"),
    ("S6.3M", "$6.3M"),
    ("Cost S1,240,000 total", "Cost $1,240,000 total"),
    ("S3 bucket", "S3 bucket"),         # a product name, not money
    ("Section S12", "Section S12"),
    ("MS6.3M", "MS6.3M"),               # inside a word
])
def test_dollar_misread_as_s_is_fixed_only_before_amounts(raw, fixed):
    assert ocr.fix_currency(raw) == fixed


def test_empty_input():
    assert layout_lines([]) == ([], [])


# --- EasyOCR engine (reader replaced) ----------------------------------------

class FakeReader:
    """Mimics easyocr.Reader.readtext output: (polygon, text, score)."""

    def __init__(self, detections):
        self.detections = detections
        self.image_shape = None

    def readtext(self, image, detail, paragraph):
        assert detail == 1 and paragraph is False
        self.image_shape = np.asarray(image).shape
        return self.detections


def page_with_size(width=600, height=800):
    document = pymupdf.open()
    return document, document.new_page(width=width, height=height)


def test_easyocr_converts_pixels_to_points_and_drops_noise():
    _document, page = page_with_size()
    engine = EasyOcrEngine(languages=["en"], dpi=144, min_line_confidence=0.3)
    poly = [[100, 200], [300, 200], [300, 220], [100, 220]]
    engine._reader = FakeReader([(poly, "Revenue grew", 0.95),
                                 (poly, "~#", 0.05)])

    result = engine.recognize(None, page)

    assert engine._reader.image_shape == (1600, 1200)  # 144 dpi, grayscale
    assert result.dropped_lines == 1
    [block] = result.blocks
    assert block["text"] == "Revenue grew\n"
    assert block["bbox"] == [50.0, 100.0, 150.0, 110.0]  # 72/144 = 0.5


def test_easyocr_region_offsets_by_the_clip_position():
    _document, page = page_with_size()
    engine = EasyOcrEngine(languages=["en"], dpi=72)
    engine._reader = FakeReader([([[0, 0], [40, 0], [40, 10], [0, 10]],
                                  "Inside", 0.9)])

    result = engine.recognize(None, page, clip=pymupdf.Rect(100, 300, 400, 500))

    assert engine._reader.image_shape == (200, 300)  # only the region
    assert result.blocks[0]["bbox"] == [100, 300, 140, 310]


def test_missing_easyocr_package_says_how_to_install(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fail_easyocr(name, *args, **kwargs):
        if name == "easyocr":
            raise ImportError("no easyocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_easyocr)
    with pytest.raises(ocr.OcrUnavailable, match="pip install"):
        EasyOcrEngine()._get_reader()


# --- Docling mapping ----------------------------------------------------------

def test_docling_items_become_blocks_and_tables_in_top_left_coordinates():
    pytest.importorskip("docling_core")
    from docling_core.types.doc import (BoundingBox, CoordOrigin,
                                        DocItemLabel, DoclingDocument, Size)
    from docling_core.types.doc.document import (ProvenanceItem, TableCell,
                                                 TableData)

    def prov(l, t, r, b):
        return ProvenanceItem(page_no=1, charspan=(0, 0), bbox=BoundingBox(
            l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.BOTTOMLEFT))

    doc = DoclingDocument(name="scan")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_text(label=DocItemLabel.TEXT, text="Intro paragraph",
                 prov=prov(50, 750, 300, 730))
    cells = [TableCell(text=text, start_row_offset_idx=r,
                       end_row_offset_idx=r + 1, start_col_offset_idx=c,
                       end_col_offset_idx=c + 1)
             for r, row in enumerate([["Year", "Visits"], ["2023", "412"]])
             for c, text in enumerate(row)]
    doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=cells),
                  prov=prov(50, 700, 300, 600))

    result = ocr.items_to_result(doc, 1)

    assert result.blocks == [{"bbox": [50, 50, 300, 70],
                              "text": "Intro paragraph\n", "confidence": None}]
    assert result.tables == [{"bbox": [50, 100, 300, 200],
                              "rows": [["Year", "Visits"], ["2023", "412"]]}]

    # For an image region, only items centred inside it are kept.
    region = ocr.items_to_result(doc, 1, clip=pymupdf.Rect(0, 90, 600, 220))
    assert region.blocks == [] and len(region.tables) == 1


# --- engine selection ---------------------------------------------------------

def test_engine_selection_follows_config(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENGINE", "none")
    assert get_ocr_engine() is None
    monkeypatch.setattr(config, "OCR_ENGINE", "easyocr")
    assert get_ocr_engine().name == "easyocr"
    assert get_ocr_engine("docling").name == "docling"
    assert get_ocr_engine() is get_ocr_engine()  # one instance: models once
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        get_ocr_engine("tesseract")


def test_mean_confidence_covers_blocks_and_tables():
    result = OcrResult(blocks=[{"confidence": 0.9}, {"confidence": None}],
                       tables=[{"confidence": 0.5}])
    assert result.mean_confidence == pytest.approx(0.7)
    assert OcrResult().mean_confidence is None
