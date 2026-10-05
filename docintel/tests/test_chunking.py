"""Chunking tests against a real parsed document (tests/fixtures).

Run from the project folder:  pytest
"""
import copy
import json
from pathlib import Path

import pytest

from docintel.chunking.pipeline import create_validated_chunks
from docintel.chunking.table_chunker import detect_header, render_row
from docintel.chunking.text_chunker import split_text_block
from docintel.parsing.cleanup import PAGE_FURNITURE, mark_page_furniture
from docintel.tokens import approximate_token_count

FIXTURE = Path(__file__).parent / "fixtures" / "sample_parsed.json"


@pytest.fixture
def parsed():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # The fixture predates furniture detection, so apply it here
    # (parse_pdf does this for new documents).
    mark_page_furniture(document["pages"])
    return document


def chunk(document, max_tokens=200, max_rows=25):
    return create_validated_chunks(
        document,
        count_tokens=approximate_token_count,
        max_tokens=max_tokens,
        max_rows_per_chunk=max_rows,
    )


def test_footer_is_marked_and_excluded(parsed):
    furniture = [
        block for page in parsed["pages"] for block in page["text_blocks"]
        if block["classification"] == PAGE_FURNITURE
    ]
    assert len(furniture) == 2
    assert all("synthetic data" in block["text"] for block in furniture)

    texts = "\n".join(c["text"] for c in chunk(parsed))
    assert "DocIntel test document - synthetic data" not in texts


def test_page_titles_are_not_furniture(parsed):
    kept = "\n".join(c["text"] for c in chunk(parsed))
    assert "DocIntel Parsing Test: Text and Tables" in kept
    assert "DocIntel Parsing Test: Continued" in kept


def test_single_page_keeps_everything_but_page_numbers():
    page = {"page_number": 1, "text_blocks": [
        {"block_number": 0, "bbox": [0, 0, 100, 10],
         "text": "Annual report\n", "classification": "OUTSIDE_TABLE"},
        {"block_number": 1, "bbox": [0, 700, 100, 710],
         "text": "Page 1\n", "classification": "OUTSIDE_TABLE"},
    ]}
    marked = mark_page_furniture([page])
    assert [b["block_number"] for b in marked] == [1]


def test_headers_detected_for_all_tables(parsed):
    found = [
        detect_header(table)
        for page in parsed["pages"] for table in page["tables"]
    ]
    assert found == [
        (["Year", "Library visits (thousands)",
          "Active members (thousands)"], 1),
        (["Category", "2024", "2025"], 1),   # year headers are allowed
        (["Program", "Location", "Participants", "Completion rate"], 1),
    ]


def test_numeric_first_row_is_not_a_header():
    table = {"rows": [["2023", "412"], ["2024", "438"]]}
    assert detect_header(table) == (None, 0)


def test_external_header_from_pymupdf_is_used():
    table = {
        "rows": [["2023", "412"]],
        "header": {"names": ["Year", "Visits"], "external": True},
    }
    assert detect_header(table) == (["Year", "Visits"], 0)


def test_rows_render_with_column_names(parsed):
    tables = [c for c in chunk(parsed) if c["content_type"] == "table"]
    program = next(c for c in tables if c["title"].startswith("Table 3"))
    assert (
        "Program: Digital skills | Location: East | Participants: 85 | "
        "Completion rate: 76%"
    ) in program["text"]


def test_render_row_skips_empty_cells_and_newlines():
    line = render_row(["Total\nall", None, " 9.6 "], ["Category", "2023", "2024"])
    assert line == "Category: Total all | 2024: 9.6"


def test_notes_attached_to_their_tables(parsed):
    tables = {c["title"]: c for c in chunk(parsed)
              if c["content_type"] == "table"}
    assert "rounded to the nearest thousand" in \
        tables["Table 1. Northport library usage"]["text"]
    assert "building repair" in \
        tables["Table 2. Northport library budget (CAD millions)"]["text"]


@pytest.mark.parametrize("max_tokens", [200, 80, 40, 25])
def test_coverage_holds_under_tight_budgets(parsed, max_tokens):
    # create_validated_chunks raises if any text or row is lost,
    # duplicated or changed, so building the chunks is the test.
    chunks = chunk(parsed, max_tokens=max_tokens)

    for c in chunks:
        if c["content_type"] == "text":
            assert c["token_count"] <= max_tokens
        elif not c["oversized"]:
            assert c["token_count"] <= max_tokens


def test_split_tables_repeat_title_and_column_names(parsed):
    tables = [c for c in chunk(parsed, max_tokens=60)
              if c["content_type"] == "table"]
    program = [c for c in tables if c["title"].startswith("Table 3")]

    assert len(program) > 1
    for part in program:
        assert part["text"].startswith(
            f"Table 3. Program participation, 2025 (part "
            f"{part['part_number']} of {len(program)})"
        )
        assert all(line.startswith("Program: ")
                   for line in part["text"].splitlines()[1:])


def test_max_rows_per_chunk(parsed):
    tables = [c for c in chunk(parsed, max_rows=1)
              if c["content_type"] == "table"]
    assert all(len(c["table"]["rows"]) == 1 for c in tables)


def test_split_text_block_without_spaces():
    text = "x" * 500
    fragments = split_text_block(text, 20, lambda s: len(s) // 5 + 2)
    assert "".join(f["text"] for f in fragments) == text
    assert all(len(f["text"]) // 5 + 2 <= 20 for f in fragments)


def test_split_prefers_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third one."
    fragments = split_text_block(text, 10, approximate_token_count)
    assert fragments[0]["text"].endswith(". ")


def test_unknown_classification_is_rejected(parsed):
    broken = copy.deepcopy(parsed)
    broken["pages"][0]["text_blocks"][0]["classification"] = "MYSTERY"
    with pytest.raises(ValueError, match="Unknown classification"):
        chunk(broken)


def test_changed_table_cell_is_caught(parsed, monkeypatch):
    import docintel.chunking.pipeline as pipeline

    original = pipeline.create_table_chunks

    def tampered(*args, **kwargs):
        chunks = original(*args, **kwargs)
        chunks[0]["table"]["rows"][0][1] = "999"
        return chunks

    monkeypatch.setattr(pipeline, "create_table_chunks", tampered)
    with pytest.raises(ValueError, match="cells changed"):
        chunk(parsed)
