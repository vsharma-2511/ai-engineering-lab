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
from pathlib import Path
import shutil
import sqlite3
import sys

from .. import config
from ..chunking.chunk_storage import initialize_chunk_storage
from ..retrieval.vector_store import initialize_vector_storage
from .processor import file_checksum
from .registry import (
    get_document_versions,
    get_latest_documents,
    initialize_registry,
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
