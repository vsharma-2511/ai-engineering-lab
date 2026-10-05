"""Turn extracted tables into retrievable chunks.

Raw cell lists embed poorly: in ["Digital skills", "East", "85", "76%"]
nothing tells the model that 85 is a participant count. Each data row is
therefore rendered with its column names:

    Program: Digital skills | Location: East | Participants: 85 | Completion rate: 76%

That rendering is the chunk text (what gets embedded and later shown to
the LLM). The exact cells are kept under chunk["table"] for validation
and exact display.
"""
import re

from .text_chunker import is_section_heading
from ..tokens import TokenCounter

TABLE_TITLE = re.compile(r"^Table\s+\d+\b", re.IGNORECASE)
TABLE_NOTE = re.compile(r"^(?:Notes?|Footnotes?|Source)\s*:", re.IGNORECASE)
YEAR = re.compile(r"^(?:19|20)\d{2}$")
NUMBER = re.compile(r"^[-+(]?[$€£]?\s*[\d.,]+\s*%?\)?$")


def clean_cell(cell) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _is_numeric(value: str) -> bool:
    return bool(NUMBER.match(value)) and any(ch.isdigit() for ch in value)


def detect_header(table: dict) -> tuple[list[str] | None, int]:
    """Return (column_names, header_rows_inside_table).

    1. If PyMuPDF found a header outside the table body, use it
       (no rows are consumed).
    2. Otherwise the first row is a header when every cell is filled and
       none is a number, except years ("Category | 2024 | 2025").
    3. Otherwise the table has no header.
    """
    rows = table["rows"]
    header = table.get("header") or {}

    if header.get("external") and header.get("names"):
        names = [clean_cell(name) for name in header["names"]]
        if all(names):
            return names, 0

    if len(rows) < 2:
        return None, 0

    first = [clean_cell(cell) for cell in rows[0]]

    if all(first) and all(
        not _is_numeric(cell) or YEAR.match(cell) for cell in first
    ):
        return first, 1

    return None, 0


def render_row(row: list, column_names: list[str] | None) -> str:
    cells = [clean_cell(cell) for cell in row]

    if column_names is None:
        return " | ".join(cell for cell in cells if cell)

    parts = []
    for index, cell in enumerate(cells):
        if not cell:
            continue
        name = column_names[index] if index < len(column_names) else ""
        parts.append(f"{name}: {cell}" if name else cell)

    return " | ".join(parts)


def find_table_notes(page: dict, table: dict) -> list[dict]:
    table_bottom = table["bbox"][3]
    notes = []

    blocks = sorted(
        page["text_blocks"],
        key=lambda block: (block["bbox"][1], block["bbox"][0]),
    )

    for block in blocks:
        if block["bbox"][1] < table_bottom:
            continue

        text = block["text"].strip()

        # Stop before entering another section or table.
        if is_section_heading(text) or TABLE_TITLE.match(text):
            break

        if block["classification"] == "INSIDE_TABLE":
            break

        if block["classification"] != "OUTSIDE_TABLE":
            continue

        if TABLE_NOTE.match(text):
            notes.append(block)

    return notes


def find_table_title(page: dict, table: dict) -> dict | None:
    table_top = table["bbox"][1]
    candidates = []

    for block in page["text_blocks"]:
        if block["classification"] != "OUTSIDE_TABLE":
            continue

        # Candidate must end above the table.
        if block["bbox"][3] > table_top:
            continue

        if TABLE_TITLE.match(block["text"].strip()):
            candidates.append(block)

    if not candidates:
        return None

    # The largest bottom coordinate is closest to the table above it.
    return max(candidates, key=lambda block: block["bbox"][3])


def _compose(
    title: str | None,
    part_number: int,
    part_count: int,
    row_lines: list[str],
    notes: list[str],
) -> str:
    heading = title or "Table"
    if part_count > 1:
        heading += f" (part {part_number} of {part_count})"

    return "\n".join([heading, *row_lines, *notes])


def _group_rows(
    row_lines: list[str],
    title: str | None,
    notes: list[str],
    count_tokens: TokenCounter,
    max_tokens: int,
    max_rows: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Greedy row groups (start, end) that fit the token budget.

    Notes are included only when every group still fits with them;
    they also remain in the text chunks, so nothing is lost.
    Placeholder part labels reserve room for "(part N of M)".
    """
    def groups_for(note_lines):
        groups, start = [], 0

        while start < len(row_lines):
            end = start + 1
            while (
                end < len(row_lines)
                and end - start < max_rows
                and count_tokens(_compose(
                    title, 99, 99, row_lines[start:end + 1], note_lines
                )) <= max_tokens
            ):
                end += 1
            groups.append((start, end))
            start = end

        return groups or [(0, 0)]

    with_notes = groups_for(notes)
    without_notes = groups_for([])

    # Keep notes unless they force extra splits.
    if notes and len(with_notes) <= len(without_notes):
        return with_notes, True

    return without_notes, False


def create_table_chunks(
    parsed_document: dict,
    count_tokens: TokenCounter,
    max_tokens: int,
    max_rows_per_chunk: int = 25,
) -> list[dict]:
    if max_rows_per_chunk <= 0:
        raise ValueError("max_rows_per_chunk must be positive")

    chunks = []

    for page in parsed_document["pages"]:
        for table in page["tables"]:
            rows = table["rows"]

            if not rows:
                raise ValueError(
                    f"Empty table {table['table_number']} "
                    f"on page {page['page_number']}"
                )

            column_names, header_row_count = detect_header(table)
            data_rows = rows[header_row_count:]
            row_lines = [render_row(row, column_names) for row in data_rows]

            title_block = find_table_title(page, table)
            title = (
                re.sub(r"\s+", " ", title_block["text"]).strip()
                if title_block else None
            )

            note_blocks = find_table_notes(page, table)
            notes = [
                re.sub(r"\s+", " ", block["text"]).strip()
                for block in note_blocks
            ]

            groups, notes_included = _group_rows(
                row_lines, title, notes,
                count_tokens, max_tokens, max_rows_per_chunk,
            )

            table_id = (
                f"{parsed_document['document_id']}"
                f":v{parsed_document['document_version']}"
                f":p{page['page_number']}"
                f":t{table['table_number']}"
            )

            for part_number, (start, end) in enumerate(groups, start=1):
                text = _compose(
                    title,
                    part_number,
                    len(groups),
                    row_lines[start:end],
                    notes if notes_included else [],
                )
                token_count = count_tokens(text)

                chunks.append({
                    "chunk_id": f"{table_id}:part{part_number}",
                    "table_id": table_id,
                    "document_id": parsed_document["document_id"],
                    "document_name": parsed_document["document_name"],
                    "document_version": parsed_document["document_version"],
                    "page_number": page["page_number"],
                    "table_number": table["table_number"],
                    "content_type": "table",
                    "bbox": table["bbox"],
                    "part_number": part_number,
                    "part_count": len(groups),
                    "header_row_count": header_row_count,
                    # Positions refer to the original rows, headers included.
                    "source_row_start": header_row_count + start,
                    "source_row_end": header_row_count + end,
                    "text": text,
                    "token_count": token_count,
                    # A single row wider than the budget cannot be split
                    # without breaking it; flag it instead of hiding it.
                    "oversized": token_count > max_tokens,
                    "table": {
                        "title": title,
                        "column_names": column_names,
                        "notes": notes,
                        # Copies, so later edits cannot silently change
                        # the parsed document that validation checks against.
                        "rows": [list(row) for row in data_rows[start:end]],
                    },
                    "title": title,
                    "title_block_number": (
                        title_block["block_number"] if title_block else None
                    ),
                    "note_block_numbers": [
                        block["block_number"] for block in note_blocks
                    ],
                })

    return chunks
