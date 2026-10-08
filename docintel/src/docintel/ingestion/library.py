"""The document library: archive processed PDFs, list and remove documents.

documents/  inbox. The watcher processes new PDFs here; once CHUNKED a
            PDF moves to archive/. NEEDS_REVIEW and FAILED stay here.
archive/    originals of processed documents, named <name>.v<version>.pdf,
            kept so `reprocess` can re-read them.

Removing a document deletes everything derived from it, for every
version: chunks, vectors, parsed JSON, registry rows and its files.

    python -m docintel.ingestion.library list
    python -m docintel.ingestion.library remove "Report.pdf"
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys

from .. import config
from ..chunking.chunk_storage import initialize_chunk_storage
from ..retrieval.vector_store import initialize_vector_storage
from .processor import file_checksum, process_document
from .registry import (
    get_document_versions,
    get_latest_document,
    get_latest_documents,
    initialize_registry,
    register_document,
    set_archive_path,
)


def archive_name(record: dict) -> str:
    path = Path(record["filename"])
    return f"{path.stem}.v{record['version']}{path.suffix}"


def current_file(record: dict) -> Path | None:
    """Where a library document's PDF is now, if it still exists.

    Resolved from the configured folders, not the absolute paths stored
    at registration, so moving the project folder breaks nothing.
    """
    if record.get("archive_path"):
        path = config.ARCHIVE_DIR / Path(record["archive_path"]).name
    else:
        path = config.DOCUMENTS_DIR / record["filename"]
    return path if path.is_file() else None


def archive_document(
    db_path: Path,
    record: dict,
    path: Path,
    archive_dir: Path | None = None,
) -> Path:
    """Move a processed library PDF out of the inbox into the archive."""
    if record.get("session_id"):
        raise ValueError("Session uploads are not archived")

    archive_dir = archive_dir or config.ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / archive_name(record)

    shutil.move(str(path), destination)
    set_archive_path(db_path, record["document_id"], destination)
    return destination


# Plain-language versions of parser warnings (see parsing/pdf_parser.py).
REVIEW_REASONS = {
    "ocr_disabled": "it is scanned and OCR is turned off",
    "ocr_failed": "OCR failed on it",
    "ocr_found_no_text": "OCR found no readable text",
    "low_ocr_confidence": "the scan is too unclear to read reliably",
    "no_text_in_document": "the document has no text",
    "little_or_no_extractable_text": "it has almost no text",
    "table_extraction_failed": "a table could not be read",
}


def parse_summary(document_id: str, parsed_dir: Path | None = None) -> dict:
    """Document type, OCR pages and review reasons from the saved parse."""
    path = (parsed_dir or config.PARSED_DIR) / f"{document_id}.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    quality = parsed.get("quality", {})
    return {
        "document_type": quality.get("document_type", "digital"),
        "ocr_pages": quality.get("ocr_pages", []),
        "review": {page["page_number"]: page["warnings"]
                   for page in parsed.get("pages", []) if page["warnings"]},
    }


def processing_message(status: str, document_id: str | None) -> str:
    """One sentence for the user about how a file was processed."""
    summary = parse_summary(document_id) if document_id else {}
    if status == "CHUNKED":
        ocr_pages = len(summary.get("ocr_pages", []))
        kind = summary.get("document_type", "digital")
        if kind == "scanned":
            return f"Ready. Scanned document: read {ocr_pages} page(s) with OCR."
        if kind == "mixed":
            return (f"Ready. Mixed document: OCR used on {ocr_pages} "
                    "page(s) with scanned content.")
        return "Ready."
    if status == "NEEDS_REVIEW":
        reasons = []
        for page, warnings in sorted(summary.get("review", {}).items()):
            for warning in warnings:
                key = warning.split(":")[0].split(" (")[0]
                if key.startswith("table_"):
                    key = "table_extraction_failed"
                reasons.append(f"page {page}: "
                               f"{REVIEW_REASONS.get(key, warning)}")
        detail = "; ".join(reasons[:3]) or "some pages could not be read"
        return f"Not searchable: {detail}."
    if status == "FAILED":
        return "Processing failed; the file is not searchable."
    return status


def add_to_library(
    file_path: Path,
    db_path: Path | None = None,
    count_tokens=None,
) -> dict:
    """Add a file to the library now, without the watcher (e.g. from the UI).

    The file is processed from a staging folder rather than the inbox, so
    a running watcher never processes it a second time. CHUNKED files go
    to the archive; NEEDS_REVIEW and FAILED ones to documents/, the same
    place the watcher leaves them.
    """
    file_path = Path(file_path)
    db_path = db_path or config.DB_PATH
    if file_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_path.name}")

    initialize_registry(db_path)
    initialize_chunk_storage(db_path)

    staging = config.UPLOADS_DIR / "library"
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / file_path.name
    if file_path.resolve() != staged.resolve():
        shutil.copy2(file_path, staged)

    try:
        result, _ = register_document(db_path, staged, file_checksum(staged))
        if result == "duplicate_content":
            return {"status": "DUPLICATE", "filename": staged.name,
                    "message": "The library already has a file with the "
                               "same contents under another name."}

        record = get_latest_document(db_path, staged.name)
        status = record["status"]
        if status == "CHUNKED" and current_file(record):
            return {"status": "CHUNKED", "filename": staged.name,
                    "message": "Already in the library."}

        if status != "CHUNKED":
            if count_tokens is None:
                from ..tokens import get_token_counter
                count_tokens = get_token_counter(config.EMBEDDING_MODEL)
            status = process_document(staged, record, count_tokens,
                                      db_path, config.PARSED_DIR)

        message = processing_message(status, record["document_id"])
        if status == "CHUNKED":
            archive_document(db_path, record, staged)
            message = message.replace(
                "Ready.", f"Added as version {record['version']}.")
        else:
            config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), config.DOCUMENTS_DIR / staged.name)
            message += " Left in documents/ for review."
        return {"status": status, "filename": staged.name,
                "message": message}
    finally:
        staged.unlink(missing_ok=True)


def find_library_document(db_path: Path, path: Path) -> dict | None:
    """The library document whose current file is `path`, if any."""
    path = Path(path)
    folder = path.parent.resolve()
    for record in get_latest_documents(db_path):
        if record["session_id"] is not None:
            continue
        if folder == config.ARCHIVE_DIR.resolve():
            if (record["archive_path"]
                    and Path(record["archive_path"]).name == path.name):
                return record
        elif folder == config.DOCUMENTS_DIR.resolve():
            if not record["archive_path"] and record["filename"] == path.name:
                return record
    return None


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _files_of(versions: list[dict]) -> set[Path]:
    """Files to remove with a document, limited to folders we manage."""
    checksums = {v["checksum"] for v in versions}
    files = set()

    for version in versions:
        if version["archive_path"]:
            files.add(config.ARCHIVE_DIR / Path(version["archive_path"]).name)

        source = Path(version["source_path"])
        if version["session_id"] and _inside(source, config.UPLOADS_DIR):
            files.add(source)

    # The inbox copy, unless it is a newer file nobody registered yet.
    if versions and versions[0]["session_id"] is None:
        inbox = config.DOCUMENTS_DIR / versions[0]["filename"]
        try:
            if inbox.is_file() and file_checksum(inbox) in checksums:
                files.add(inbox)
        except OSError:
            pass

    return {path for path in files if path.is_file()}


def delete_document(
    db_path: Path,
    key: str,
    parsed_dir: Path | None = None,
) -> dict | None:
    """Delete every version of a document and everything derived from it.

    `key` is the registry key: the filename for library documents (see
    registry.registry_key for session uploads). Returns a summary, or
    None if nothing is registered under `key`.
    """
    parsed_dir = parsed_dir or config.PARSED_DIR
    initialize_registry(db_path)
    initialize_chunk_storage(db_path)
    initialize_vector_storage(db_path)

    versions = get_document_versions(db_path, key)
    if not versions:
        return None

    ids = [version["document_id"] for version in versions]
    files = _files_of(versions)
    marks = ",".join("?" * len(ids))

    # Database first: once these rows are gone nothing can be retrieved,
    # even if removing a file below fails.
    with sqlite3.connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        vectors = connection.execute(
            f"""
            DELETE FROM chunk_vectors WHERE chunk_id IN (
                SELECT chunk_id FROM chunks WHERE document_id IN ({marks})
            )
            """,
            ids,
        ).rowcount
        chunks = connection.execute(
            f"DELETE FROM chunks WHERE document_id IN ({marks})", ids,
        ).rowcount
        connection.execute(
            f"DELETE FROM session_documents WHERE document_id IN ({marks})",
            ids,
        )
        connection.execute(
            f"DELETE FROM documents WHERE document_id IN ({marks})", ids,
        )

    removed = []
    for path in [*files, *(parsed_dir / f"{i}.json" for i in ids)]:
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Could not remove {path}: {exc}", flush=True)

    return {
        "key": key,
        "versions": len(versions),
        "chunks": chunks,
        "vectors": vectors,
        "files_removed": removed,
    }


def remove_missing_documents(db_path: Path) -> list[dict]:
    """Delete library documents whose PDF was removed by hand.

    Run at watcher startup to catch deletions made while it was off.
    Skipped when the archive folder is missing entirely, which points to
    a misconfiguration rather than deliberate deletions.
    """
    if not config.ARCHIVE_DIR.is_dir():
        return []

    summaries = []
    for record in get_latest_documents(db_path):
        if record["session_id"] is None and current_file(record) is None:
            summary = delete_document(db_path, record["filename"])
            if summary:
                summaries.append(summary)
    return summaries


def describe(summary: dict) -> str:
    return (
        f"{summary['key']} | versions={summary['versions']} | "
        f"chunks={summary['chunks']} | vectors={summary['vectors']} | "
        f"files={len(summary['files_removed'])}"
    )


def list_library(db_path: Path) -> list[dict]:
    initialize_registry(db_path)
    initialize_chunk_storage(db_path)

    with sqlite3.connect(db_path) as connection:
        counts = dict(connection.execute(
            "SELECT document_id, COUNT(*) FROM chunks GROUP BY document_id"
        ).fetchall())

    rows = []
    for record in get_latest_documents(db_path):
        if record["session_id"] is not None:
            continue
        location = current_file(record)
        rows.append({
            **record,
            "document_type": parse_summary(record["document_id"]).get(
                "document_type", ""),
            "chunks": counts.get(record["document_id"], 0),
            "location": (
                str(location.relative_to(config.PROJECT_ROOT))
                if location and _inside(location, config.PROJECT_ROOT)
                else str(location or "missing")
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Show library documents")
    remove = commands.add_parser(
        "remove", help="Delete a document and all its data"
    )
    remove.add_argument("filename")
    remove.add_argument("--yes", action="store_true",
                        help="Do not ask for confirmation")
    args = parser.parse_args()

    if args.command == "list":
        rows = list_library(config.DB_PATH)
        if not rows:
            print("The library is empty.")
        for row in rows:
            print(f"{row['filename']}  v{row['version']}  {row['status']}  "
                  f"chunks={row['chunks']}  {row['location']}")
        return 0

    initialize_registry(config.DB_PATH)
    versions = get_document_versions(config.DB_PATH, args.filename)
    if not versions:
        print(f"Not in the library: {args.filename}")
        return 2

    if not args.yes:
        answer = input(
            f"Delete {args.filename} ({len(versions)} version(s)) with its "
            "chunks, vectors and files? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    print("REMOVED:", describe(delete_document(config.DB_PATH, args.filename)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
