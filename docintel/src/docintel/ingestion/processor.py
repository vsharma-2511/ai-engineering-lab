"""Parse, chunk and store one registered document version.

Shared by the watcher and the reprocess command so both behave the same.
"""
from pathlib import Path
import hashlib

from .. import config
from ..chunking.chunk_storage import save_chunks
from ..chunking.pipeline import create_validated_chunks
from ..parsing.pdf_parser import parse_pdf, save_parse_result
from ..tokens import TokenCounter
from .registry import update_document_status


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def process_document(
    path: Path,
    record: dict,
    count_tokens: TokenCounter,
    db_path: Path = config.DB_PATH,
    parsed_dir: Path = config.PARSED_DIR,
) -> str:
    """Return the final status and print a one-line summary."""
    document_id = record["document_id"]
    update_document_status(db_path, document_id, "PROCESSING")

    try:
        parsed = parse_pdf(path, document_id, record["version"])
        output_path = save_parse_result(parsed, parsed_dir)

        review_pages = parsed["quality"]["pages_needing_review"]

        if review_pages:
            update_document_status(db_path, document_id, "NEEDS_REVIEW")
            print(
                f"NEEDS_REVIEW: {path.name} | "
                f"review_pages={review_pages} | result={output_path}",
                flush=True,
            )
            return "NEEDS_REVIEW"

        chunks = create_validated_chunks(
            parsed,
            count_tokens=count_tokens,
            max_tokens=config.CHUNK_MAX_TOKENS,
            max_rows_per_chunk=config.TABLE_MAX_ROWS_PER_CHUNK,
        )

        if file_checksum(path) != record["checksum"]:
            raise ValueError(
                "File changed during processing; chunks were not saved. "
                "Rescan the file."
            )

        save_chunks(db_path, chunks)
        update_document_status(db_path, document_id, "CHUNKED")

        oversized = [chunk["chunk_id"] for chunk in chunks
                     if chunk.get("oversized")]
        warning = f" | oversized={len(oversized)}" if oversized else ""

        print(
            f"CHUNKED: {path.name} | "
            f"type={parsed['quality']['document_type']} | "
            f"pages={parsed['quality']['total_pages']} | "
            f"chunks={len(chunks)}{warning} | result={output_path}",
            flush=True,
        )
        return "CHUNKED"

    except Exception as exc:
        update_document_status(db_path, document_id, "FAILED", str(exc))
        print(f"FAILED: {path.name} | {exc}", flush=True)
        return "FAILED"
