"""Page classification and OCR routing in the parser, on generated PDFs.

OCR itself is replaced by a fake engine; see test_ocr.py for the
engines. Run from the project folder:  pytest
"""
import pymupdf
import pytest

from docintel import config
from docintel.chunking.pipeline import create_validated_chunks
from docintel.parsing.ocr import OcrResult
from docintel.parsing.page_analysis import analyze_page, document_type
from docintel.parsing.pdf_parser import parse_pdf
from docintel.tokens import approximate_token_count

PARAGRAPH = ("Northport expanded its library services between 2023 and 2025. "
             "Visits are counted at the entrance.")


def picture(width=400, height=300):
    """PNG bytes of a grey picture (pixels only, no text layer)."""
    pixmap = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, width, height),
                            False)
    pixmap.clear_with(200)
    return pixmap.tobytes("png")


def build_pdf(path, pages):
    """pages: list of lists of ("text", ...), ("image", rect), ("lines", n)."""
    document = pymupdf.open()
    for items in pages:
        page = document.new_page(width=600, height=800)
        for kind, *args in items:
            if kind == "text":
                page.insert_textbox(pymupdf.Rect(50, args[1] if len(args) > 1
                                                 else 50, 550, 780),
                                    args[0], fontsize=11)
            elif kind == "image":
                # Same proportions as the target, so it fills it exactly.
                rect = args[0]
                page.insert_image(rect, stream=picture(int(rect.width),
                                                       int(rect.height)))
            elif kind == "lines":
                for i in range(args[0]):
                    page.draw_line((50, 100 + i * 5), (550, 100 + i * 5))
    document.save(path)
    return path


def analyze(path):
    with pymupdf.open(path) as document:
        return [analyze_page(page) for page in document]


FULL_PAGE = pymupdf.Rect(0, 0, 600, 800)


# --- classification ------------------------------------------------------------

@pytest.mark.parametrize("items, kind", [
    ([("text", PARAGRAPH)], "text"),
    ([("image", FULL_PAGE)], "scanned"),
    ([("text", PARAGRAPH), ("image", pymupdf.Rect(50, 300, 550, 700))], "mixed"),
    ([], "blank"),
    # A small logo does not make a typed page "mixed".
    ([("text", PARAGRAPH), ("image", pymupdf.Rect(500, 20, 560, 60))], "text"),
    # Text drawn as vector outlines: no text layer, but something to OCR.
    ([("lines", 60)], "scanned"),
    # A few rules or boxes on an empty page are not content.
    ([("lines", 3)], "blank"),
])
def test_page_kinds(tmp_path, items, kind):
    [analysis] = analyze(build_pdf(tmp_path / "doc.pdf", [items]))
    assert analysis.kind == kind


def test_mixed_page_lists_the_image_region_to_ocr(tmp_path):
    region = pymupdf.Rect(50, 300, 550, 700)
    [analysis] = analyze(build_pdf(tmp_path / "doc.pdf",
                                   [[("text", PARAGRAPH), ("image", region)]]))
    assert analysis.ocr_regions == [region]
    assert analysis.image_coverage == pytest.approx(region.get_area() / 480000)


def test_scan_with_a_text_layer_is_read_directly(tmp_path):
    # Already-OCR'd scans carry text over the image: no second OCR.
    path = build_pdf(tmp_path / "doc.pdf", [[("image", FULL_PAGE),
                                             ("text", PARAGRAPH, 60)]])
    assert analyze(path)[0].kind == "text"


@pytest.mark.parametrize("kinds, expected", [
    (["text", "text", "blank"], "digital"),
    (["scanned", "blank", "scanned"], "scanned"),
    (["text", "scanned"], "mixed"),
    (["text", "mixed"], "mixed"),
    (["blank"], "digital"),
])
def test_document_type(kinds, expected):
    assert document_type(kinds) == expected


# --- parser routing with a fake OCR engine --------------------------------------

class FakeOcr:
    """Returns prepared OCR output and records what it was asked to read."""

    name = "fake-ocr"

    def __init__(self, result=None, error=None):
        self.result = result or OcrResult(engine=self.name)
        self.error = error
        self.calls = []

    def recognize(self, pdf_path, page, clip=None):
        self.calls.append((page.number + 1, clip))
        if self.error:
            raise self.error
        return self.result


SCANNED_TABLE = OcrResult(
    engine="fake-ocr",
    blocks=[{"bbox": [50, 40, 400, 60], "text": "Table 1. Library usage\n",
             "confidence": 0.95}],
    tables=[{"bbox": [50, 80, 500, 160], "confidence": 0.9, "rows": [
        ["Year", "Visits", "Members"],
        ["2023", "412", "68"],
        ["2024", "438", "71"],
    ]}],
)


def parse(path, engine):
    return parse_pdf(path, "doc-1", 1, ocr_engine=engine)


def test_scanned_page_is_read_with_ocr_and_chunks_like_typed_text(tmp_path):
    engine = FakeOcr(SCANNED_TABLE)
    parsed = parse(build_pdf(tmp_path / "scan.pdf",
                             [[("image", FULL_PAGE)]]), engine)

    [page] = parsed["pages"]
    assert engine.calls == [(1, None)]          # whole page, once
    assert page["page_type"] == "scanned" and page["method"] == "ocr"
    assert page["warnings"] == []
    assert page["text_blocks"][0]["source"] == "ocr"
    assert page["tables"][0]["source"] == "ocr"
    assert parsed["quality"]["document_type"] == "scanned"
    assert parsed["parser"] == "pymupdf+fake-ocr"

    # Downstream does not care that the page was scanned.
    chunks = create_validated_chunks(parsed, approximate_token_count, 200)
    table_text = next(c["text"] for c in chunks
                      if c["content_type"] == "table")
    assert "Year: 2023 | Visits: 412 | Members: 68" in table_text


def test_mixed_page_keeps_text_layer_and_ocrs_only_the_image(tmp_path):
    region = pymupdf.Rect(50, 300, 550, 700)
    ocr_block = OcrResult(engine="fake-ocr", blocks=[{
        "bbox": [60, 320, 500, 340], "text": "Scanned appendix note\n",
        "confidence": 0.9}])
    engine = FakeOcr(ocr_block)

    parsed = parse(build_pdf(tmp_path / "mixed.pdf",
                             [[("text", PARAGRAPH), ("image", region)]]),
                   engine)

    [page] = parsed["pages"]
    assert engine.calls == [(1, region)]
    assert page["method"] == "embedded_text+ocr"
    sources = {b["source"] for b in page["text_blocks"]}
    assert sources == {"text_layer", "ocr"}
    numbers = [b["block_number"] for b in page["text_blocks"]]
    assert len(numbers) == len(set(numbers))    # unique per page
    assert "Northport expanded" in page["text"]
    assert "Scanned appendix note" in page["text"]
    assert parsed["quality"]["document_type"] == "mixed"
    create_validated_chunks(parsed, approximate_token_count, 200)


def test_typed_document_never_touches_ocr(tmp_path):
    engine = FakeOcr()
    parsed = parse(build_pdf(tmp_path / "typed.pdf",
                             [[("text", PARAGRAPH)], [("text", PARAGRAPH)]]),
                   engine)
    assert engine.calls == []
    assert parsed["parser"] == "pymupdf"
    assert parsed["quality"]["document_type"] == "digital"


@pytest.mark.parametrize("engine, warning", [
    (None, "ocr_disabled"),
    (FakeOcr(error=RuntimeError("model download failed")), "ocr_failed"),
    (FakeOcr(OcrResult(engine="fake-ocr")), "ocr_found_no_text"),
    (FakeOcr(OcrResult(engine="fake-ocr", blocks=[{
        "bbox": [50, 50, 300, 70], "text": "blurry words\n",
        "confidence": 0.31}])), "low_ocr_confidence"),
])
def test_unreadable_scans_are_flagged_for_review(tmp_path, engine, warning):
    parsed = parse(build_pdf(tmp_path / "scan.pdf",
                             [[("image", FULL_PAGE)]]), engine)
    assert parsed["pages"][0]["warnings"][0].startswith(warning)
    assert parsed["quality"]["pages_needing_review"] == [1]


def test_blank_pages_are_fine_unless_the_whole_document_is_blank(tmp_path):
    with_blank = parse(build_pdf(tmp_path / "a.pdf",
                                 [[("text", PARAGRAPH)], []]), FakeOcr())
    assert with_blank["quality"]["pages_needing_review"] == []
    assert with_blank["pages"][1]["page_type"] == "blank"

    all_blank = parse(build_pdf(tmp_path / "b.pdf", [[], []]), FakeOcr())
    assert all_blank["pages"][0]["warnings"] == ["no_text_in_document"]


def test_configured_engine_loads_only_when_a_page_needs_it(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(config, "OCR_ENGINE", "no-such-engine")
    typed = parse_pdf(build_pdf(tmp_path / "typed.pdf",
                                [[("text", PARAGRAPH)]]), "d", 1)
    assert typed["quality"]["pages_needing_review"] == []   # never loaded

    scanned = parse_pdf(build_pdf(tmp_path / "scan.pdf",
                                  [[("image", FULL_PAGE)]]), "d", 1)
    assert "Unknown OCR engine" in scanned["pages"][0]["warnings"][0]


def test_ocr_block_beside_an_ocr_table_is_text_not_an_error(tmp_path):
    """A chart label partly overlapping an OCR table's box must not fail
    the document (it did: "Review overlapping block 14 on page 2")."""
    result = OcrResult(engine="fake-ocr", tables=[{
        "bbox": [100, 100, 400, 200], "confidence": 0.9,
        "rows": [["Region", "Revenue"], ["East", "5.1"], ["West", "3.2"]]}],
        blocks=[{"bbox": [380, 190, 460, 210], "text": "84\n",
                 "confidence": 0.9}])
    parsed = parse(build_pdf(tmp_path / "scan.pdf", [[("image", FULL_PAGE)]]),
                   FakeOcr(result))

    label = next(b for b in parsed["pages"][0]["text_blocks"]
                 if b["text"] == "84\n")
    # Kept as text (here even recognised as a page-number-like edge
    # block), never a PARTIAL_OVERLAP that fails the document.
    assert label["classification"] != "PARTIAL_OVERLAP"
    create_validated_chunks(parsed, approximate_token_count, 200)
