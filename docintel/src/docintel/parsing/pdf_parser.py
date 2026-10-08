"""Read a PDF into pages of text blocks and tables.

Each page is first classified (parsing/page_analysis.py):

  text     the text layer is read with PyMuPDF (exact, fast)
  scanned  the whole page goes through OCR (parsing/ocr.py)
  mixed    the text layer is read, plus OCR for large images on the page
  blank    nothing to read (not a problem in itself)

Whatever the source, every page comes out in the same shape (text
blocks with positions, tables with rows), so chunking and everything
after it do not care whether a page was typed or scanned. Each block
records its "source" ("text_layer" or "ocr"), and the document records
its type: digital, scanned or mixed.
"""
import json
import os
from pathlib import Path
import tempfile

import pymupdf

from .. import config
from .cleanup import mark_page_furniture
from .ocr import OcrResult, get_ocr_engine
from .page_analysis import MIN_USEFUL_CHARACTERS, analyze_page, document_type

DEFAULT_ENGINE = object()  # use the engine from config


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


def _table_record(number: int, page_number: int, bbox, rows, header,
                  source: str, warnings: list[str]) -> dict:
    if not rows:
        warnings.append(f"table_{number}_has_no_extracted_rows")
    elif any(all(cell is None or not str(cell).strip() for cell in row)
             for row in rows):
        warnings.append(f"table_{number}_has_empty_row")

    return {
        "table_number": number,
        "page_number": page_number,
        "bbox": list(bbox),
        "rows": rows,
        "header": header,
        "row_count": len(rows),
        "column_count": max((len(row) for row in rows), default=0),
        "source": source,
    }


def _read_text_layer(page, warnings: list[str]) -> tuple[list, list]:
    """Tables and text blocks from the PDF's own text layer."""
    tables = []
    try:
        found_tables = page.find_tables()

        for table_index, table in enumerate(found_tables.tables, start=1):
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
            tables.append(_table_record(
                table_index, page.number + 1, table.bbox, rows, header,
                "text_layer", warnings,
            ))

    except Exception as exc:
        warnings.append(f"table_extraction_failed: {exc}")

    blocks = []
    for x0, y0, x1, y1, text, number, block_type in page.get_text("blocks"):
        if block_type == 0:
            blocks.append({"block_number": number, "bbox": [x0, y0, x1, y1],
                           "text": text, "source": "text_layer"})
    return tables, blocks


def _add_ocr(result: OcrResult, page, tables: list, blocks: list,
             warnings: list[str]) -> None:
    """Append OCR blocks and tables, numbered after the existing ones."""
    next_block = max((b["block_number"] for b in blocks), default=-1) + 1
    for offset, block in enumerate(result.blocks):
        blocks.append({
            "block_number": next_block + offset,
            "bbox": [round(v, 2) for v in block["bbox"]],
            "text": block["text"],
            "source": "ocr",
            "confidence": block.get("confidence"),
        })
    for table in result.tables:
        tables.append(_table_record(
            len(tables) + 1, page.number + 1, table["bbox"], table["rows"],
            None, "ocr", warnings,
        ))


def _run_ocr(engine, path: Path, page, analysis, warnings: list[str]):
    """OCR the whole page (scanned) or its image regions (mixed)."""
    if engine is None:
        warnings.append("ocr_disabled: page needs OCR "
                        "(set DOCINTEL_OCR_ENGINE)")
        return None

    clips = [None] if analysis.kind == "scanned" else analysis.ocr_regions
    combined = OcrResult(engine=engine.name)
    for clip in clips:
        try:
            result = engine.recognize(path, page, clip)
        except Exception as exc:  # missing package, model download, ...
            warnings.append(f"ocr_failed ({engine.name}): {exc}")
            return None
        combined.blocks += result.blocks
        combined.tables += result.tables
        combined.dropped_lines += result.dropped_lines

    if analysis.kind == "scanned":
        if not combined.blocks and not combined.tables:
            warnings.append("ocr_found_no_text")
        elif (combined.mean_confidence is not None
              and combined.mean_confidence < config.OCR_MIN_PAGE_CONFIDENCE):
            warnings.append(
                f"low_ocr_confidence: {combined.mean_confidence:.2f}")
    return combined


def parse_pdf(
    path: Path,
    document_id: str,
    version: int,
    ocr_engine=DEFAULT_ENGINE,
) -> dict:
    """Parse a PDF. `ocr_engine`: an engine from parsing.ocr, None for no
    OCR, or omitted for the configured one (loaded only if a page needs it).
    """
    pages = []
    engine = None
    engine_loaded = ocr_engine is not DEFAULT_ENGINE
    if engine_loaded:
        engine = ocr_engine

    with pymupdf.open(path) as pdf:
        if not pdf.is_pdf:
            raise ValueError(f"Not a PDF: {path.name}")

        if pdf.page_count == 0:
            raise ValueError(f"PDF contains no pages: {path.name}")

        for page_index, page in enumerate(pdf):
            analysis = analyze_page(page)
            warnings = []
            tables, blocks = [], []
            ocr = None

            if analysis.kind in {"text", "mixed"}:
                tables, blocks = _read_text_layer(page, warnings)

            if analysis.kind in {"scanned", "mixed"}:
                if not engine_loaded:
                    try:
                        engine = get_ocr_engine()
                    except ValueError as exc:
                        warnings.append(f"ocr_failed: {exc}")
                    engine_loaded = True
                ocr = _run_ocr(engine, path, page, analysis, warnings)
                if ocr is not None:
                    _add_ocr(ocr, page, tables, blocks, warnings)

            # For our simple layout: top to bottom, then left to right.
            blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
            for block in blocks:
                block["classification"], block["table_number"] = (
                    classify_block(block["bbox"], tables))
                # OCR builds its tables and text blocks from different
                # words, so a partial overlap is only geometry (a label
                # beside a table), never duplicated text: keep it as text.
                # Text-layer overlaps stay errors that need review.
                if (block["source"] == "ocr"
                        and block["classification"] == "PARTIAL_OVERLAP"):
                    block["classification"], block["table_number"] = (
                        "OUTSIDE_TABLE", None)

            if analysis.kind == "scanned":
                text = "\n\n".join(b["text"].strip() for b in blocks).strip()
                method = "ocr"
            else:
                text = page.get_text("text", sort=True).strip()
                method = {"text": "embedded_text", "blank": "none",
                          "mixed": "embedded_text+ocr"}[analysis.kind]
                if analysis.kind == "mixed" and blocks and any(
                        b["source"] == "ocr" for b in blocks):
                    text = "\n\n".join([text] + [
                        b["text"].strip() for b in blocks
                        if b["source"] == "ocr"])

            if analysis.kind == "text" and len(text) < MIN_USEFUL_CHARACTERS:
                warnings.append("little_or_no_extractable_text")

            pages.append({
                "page_number": page_index + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "page_type": analysis.kind,
                "page_type_reason": analysis.reason,
                "image_coverage": round(analysis.image_coverage, 3),
                "text": text,
                "method": method,
                "character_count": len(text),
                "ocr": None if ocr is None else {
                    "engine": ocr.engine,
                    "regions": (1 if analysis.kind == "scanned"
                                else len(analysis.ocr_regions)),
                    "blocks": len(ocr.blocks),
                    "tables": len(ocr.tables),
                    "mean_confidence": (None if ocr.mean_confidence is None
                                        else round(ocr.mean_confidence, 3)),
                    "dropped_lines": ocr.dropped_lines,
                },
                "tables": tables,
                "text_blocks": blocks,
                "warnings": warnings,
            })

    # A document of blank pages has nothing to index: say so.
    if not any(page["text"] for page in pages):
        pages[0]["warnings"].append("no_text_in_document")

    furniture = mark_page_furniture(pages)
    kinds = [page["page_type"] for page in pages]
    ocr_pages = [p["page_number"] for p in pages if p["ocr"] is not None]
    used_engine = next((p["ocr"]["engine"] for p in pages if p["ocr"]), None)

    return {
        "document_id": document_id,
        "document_name": path.name,
        "document_version": version,
        "parser": "pymupdf" + (f"+{used_engine}" if used_engine else ""),
        "pages": pages,
        "quality": {
            "total_pages": len(pages),
            "document_type": document_type(kinds),
            "page_types": {kind: [p["page_number"] for p in pages
                                  if p["page_type"] == kind]
                           for kind in ("text", "scanned", "mixed", "blank")
                           if kind in kinds},
            "ocr_pages": ocr_pages,
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