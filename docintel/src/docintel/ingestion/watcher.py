"""Watch the documents folder and process stable new or changed PDFs.

Processed (CHUNKED) PDFs move to archive/. Deleting a document's PDF
from documents/ or archive/ removes the document with all its data.

Run from the project folder:      python -m docintel.ingestion.watcher
or from one folder above it:      python -m docintel.src.docintel.ingestion.watcher
"""
from pathlib import Path
import time

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .. import config
from ..chunking.chunk_storage import initialize_chunk_storage
from ..tokens import get_token_counter
from .library import (
    archive_document,
    current_file,
    delete_document,
    describe,
    find_library_document,
    remove_missing_documents,
)
from .processor import file_checksum, process_document
from .registry import (
    get_latest_document,
    initialize_registry,
    register_document,
)

# Each event postpones the check for this path.
pending: dict[Path, float] = {}
# Paths that disappeared; checked against the registry in the main loop.
removed_paths: set[Path] = set()


def queue_candidate(path: Path) -> None:
    # Only the inbox is processed; archive/ events are ours or deletions.
    if (
        path.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and path.parent.resolve() == config.DOCUMENTS_DIR.resolve()
    ):
        pending[path] = time.monotonic()


def queue_removal(path: Path) -> None:
    if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
        removed_paths.add(path)


def process_removals() -> None:
    while removed_paths:
        path = removed_paths.pop()
        record = find_library_document(config.DB_PATH, path)
        # Our own archiving also "removes" the inbox file; the document
        # then still has a current file and is left alone.
        if record is None or current_file(record) is not None:
            continue
        summary = delete_document(config.DB_PATH, record["filename"])
        if summary:
            print(f"REMOVED: {describe(summary)}", flush=True)


def archive(record: dict, path: Path) -> None:
    destination = archive_document(config.DB_PATH, record, path)
    print(f"ARCHIVED: {path.name} -> "
          f"{destination.relative_to(config.PROJECT_ROOT)}", flush=True)


def process_ready_candidates(count_tokens) -> None:
    now = time.monotonic()

    for path, last_event_at in list(pending.items()):
        if now - last_event_at < config.WATCHER_QUIET_SECONDS:
            continue

        if not path.is_file():
            pending.pop(path, None)
            continue

        try:
            before = path.stat()
            checksum = file_checksum(path)
            after = path.stat()

            # The file changed while we were reading it: retry later.
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                pending[path] = time.monotonic()
                continue

            result, _ = register_document(config.DB_PATH, path, checksum)

        except OSError as exc:
            print(f"Could not read {path.name}: {exc}", flush=True)
            pending[path] = time.monotonic()
            continue

        if result == "duplicate_content":
            print(f"Duplicate content: {path.name}", flush=True)
            pending.pop(path, None)
            continue

        record = get_latest_document(config.DB_PATH, path.name)

        if record is None or record["checksum"] != checksum:
            print(f"Registry mismatch for {path.name}", flush=True)
            pending.pop(path, None)
            continue

        # A startup scan or another file event must not parse it repeatedly.
        # Use `python -m ...ingestion.reprocess` to force a rerun.
        if record["status"] in {"CHUNKED", "NEEDS_REVIEW"}:
            print(
                f"Already handled: {path.name} ({record['status']})",
                flush=True,
            )
            # Processed before archiving existed, or dropped in again.
            if record["status"] == "CHUNKED":
                archive(record, path)
            pending.pop(path, None)
            continue

        status = process_document(path, record, count_tokens,
                                  config.DB_PATH, config.PARSED_DIR)
        if status == "CHUNKED":
            archive(record, path)
        pending.pop(path, None)


class DocumentEventHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            queue_candidate(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            queue_candidate(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            queue_removal(Path(event.src_path))
            queue_candidate(Path(event.dest_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            queue_removal(Path(event.src_path))


def main() -> None:
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    initialize_registry(config.DB_PATH)
    initialize_chunk_storage(config.DB_PATH)

    # Catch PDFs deleted while the watcher was not running.
    for summary in remove_missing_documents(config.DB_PATH):
        print(f"REMOVED (file missing): {describe(summary)}", flush=True)

    count_tokens = get_token_counter(config.EMBEDDING_MODEL)

    observer = Observer()
    for folder in (config.DOCUMENTS_DIR, config.ARCHIVE_DIR):
        observer.schedule(
            DocumentEventHandler(), str(folder), recursive=False,
        )
    observer.start()

    # Catch files that arrived while the watcher was not running.
    for path in config.DOCUMENTS_DIR.iterdir():
        queue_candidate(path)

    print(f"Watching: {config.DOCUMENTS_DIR}", flush=True)
    print(f"Archive:  {config.ARCHIVE_DIR}", flush=True)
    print(f"Registry: {config.DB_PATH}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            process_ready_candidates(count_tokens)
            process_removals()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
