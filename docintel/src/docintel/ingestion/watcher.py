"""Watch the documents folder and process stable new or changed PDFs.

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
from .processor import file_checksum, process_document
from .registry import (
    get_latest_document,
    initialize_registry,
    register_document,
)

# Each event postpones the check for this path.
pending: dict[Path, float] = {}


def queue_candidate(path: Path) -> None:
    if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
        pending[path] = time.monotonic()


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
            pending.pop(path, None)
            continue

        process_document(path, record, count_tokens)
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
            queue_candidate(Path(event.dest_path))


def main() -> None:
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_registry(config.DB_PATH)
    initialize_chunk_storage(config.DB_PATH)

    count_tokens = get_token_counter(config.EMBEDDING_MODEL)

    observer = Observer()
    observer.schedule(
        DocumentEventHandler(),
        str(config.DOCUMENTS_DIR),
        recursive=False,
    )
    observer.start()

    # Catch files that arrived while the watcher was not running.
    for path in config.DOCUMENTS_DIR.iterdir():
        queue_candidate(path)

    print(f"Watching: {config.DOCUMENTS_DIR}", flush=True)
    print(f"Registry: {config.DB_PATH}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            process_ready_candidates(count_tokens)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
