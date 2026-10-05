import json
import os
from pathlib import Path
import tempfile

import pymupdf

from .cleanup import mark_page_furniture


MIN_USEFUL_CHARACTERS = 20


def classify_block(block_bbox, tables):
    block_rect = pymupdf.Rect(block_bbox)

    for table in tables:
        table_rect = pymupdf.Rect(table["bbox"])

        if table_rect.contains(block_rect):
            return "INSIDE_TABLE", table["table_number"]

    # Check overlap only after checking containment against all tables.
    for table in tables:
        table_rect = pymupdf.Rect(table["bbox"])
        intersection = block_rect & table_rect

        if intersection.get_area() > 0:
            return "PARTIAL_OVERLAP", table["table_number"]

    return "OUTSIDE_TABLE", None

def parse_pdf(path: Path, document_id: str, version: int) -> dict:
    pages = []

    with pymupdf.open(path) as pdf:
        if not pdf.is_pdf:
            raise ValueError(f"Not a PDF: {path.name}")

        if pdf.page_count == 0:
            raise ValueError(f"PDF contains no pages: {path.name}")

        for page_index, page in enumerate(pdf):
            text = page.get_text("text", sort=True).strip()

            warnings = []
            tables = []

            try:
                found_tables = page.find_tables()

                for table_index, table in enumerate(
                    found_tables.tables,
                    start=1,
                ):
                    rows = table.extract()

                    # PyMuPDF's guess at the header. "external" means the
                    # header sits above the table and is not part of rows.
                    try:
                        header = {
                            "names": list(table.header.names),
                            "external": bool(table.header.external),
                        }
                    except Exception:
                        header = None

                    # Preserve the cells exactly as extracted. Do not guess
                    # that the first row is a header yet.
                    tables.append({
                        "table_number": table_index,
                        "page_number": page_index + 1,
                        "bbox": list(table.bbox),
                        "rows": rows,
                        "header": header,
                        "row_count": len(rows),
                        "column_count": max(
                            (len(row) for row in rows),
                            default=0,
                        ),
                    })

                    if not rows:
                        warnings.append(
                            f"table_{table_index}_has_no_extracted_rows"
                        )
                    elif any(
                        all(cell is None or not str(cell).strip()
                            for cell in row)
                        for row in rows
                    ):
                        warnings.append(
                            f"table_{table_index}_has_empty_row"
                        )

            except Exception as exc:
                warnings.append(f"table_extraction_failed: {exc}")

            blocks = page.get_text("blocks")

            # For our simple layout: top to bottom, then left to right.
            blocks = sorted(blocks, key=lambda block: (block[1], block[0]))

            classified_blocks = []

            for block in blocks:
                x0, y0, x1, y1, block_text, block_number, block_type = block

                if block_type != 0:
                    continue

                classification, table_number = classify_block(
                    (x0, y0, x1, y1),
                    tables,
                )

                classified_blocks.append({
                    "block_number": block_number,
                    "bbox": [x0, y0, x1, y1],
                    "text": block_text,
                    "classification": classification,
                    "table_number": table_number,
                })

            if len(text) < MIN_USEFUL_CHARACTERS:
                warnings.append("little_or_no_extractable_text")

            pages.append({
                "page_number": page_index + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "text": text,
                "method": "embedded_text",
                "character_count": len(text),
                "tables": tables,
                "text_blocks": classified_blocks,
                "warnings": warnings,
            })

    furniture = mark_page_furniture(pages)

    return {
        "document_id": document_id,
        "document_name": path.name,
        "document_version": version,
        "parser": "pymupdf",
        "pages": pages,
        "quality": {
            "total_pages": len(pages),
            "furniture_blocks_removed": len(furniture),
            "pages_with_text": sum(bool(p["text"]) for p in pages),
            "pages_with_tables": [
                p["page_number"] for p in pages if p["tables"]
            ],
            "pages_needing_review": [
                p["page_number"] for p in pages if p["warnings"]
            ],
        },
    }

def save_parse_result(result: dict, output_dir: Path) -> Path:
    """Write the complete result atomically under its document ID."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{result['document_id']}.json"

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(result, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # Rename after the file is closed (required on Windows).
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination