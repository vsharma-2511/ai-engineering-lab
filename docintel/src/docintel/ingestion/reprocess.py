"""Re-parse and re-chunk the latest version of every registered document.

Use after changing parsing or chunking code. The watcher skips documents
that are already CHUNKED, so it will not pick up such changes by itself.

    python -m docintel.ingestion.reprocess            # all documents
    python -m docintel.ingestion.reprocess --only "Report.pdf"
"""
import argparse
from pathlib import Path

from .. import config
from ..chunking.chunk_storage import initialize_chunk_storage
from ..tokens import get_token_counter
from .library import current_file
from .processor import file_checksum, process_document
from .registry import get_latest_documents, initialize_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Reprocess just this filename")
    args = parser.parse_args()

    initialize_registry(config.DB_PATH)
    initialize_chunk_storage(config.DB_PATH)
    count_tokens = get_token_counter(config.EMBEDDING_MODEL)

    records = get_latest_documents(config.DB_PATH)
    if args.only:
        records = [r for r in records if r["filename"] == args.only]

    if not records:
        print("No registered documents to reprocess.")
        return

    summary = {}

    for record in records:
        # The archive, then the inbox, then the path it was registered at.
        path = current_file(record) or Path(record["source_path"])

        if not path.is_file():
            print(f"SKIPPED: {record['filename']} | source file not found")
            summary["SKIPPED"] = summary.get("SKIPPED", 0) + 1
            continue

        if file_checksum(path) != record["checksum"]:
            print(
                f"SKIPPED: {record['filename']} | file differs from the "
                "registered version; let the watcher register it first"
            )
            summary["SKIPPED"] = summary.get("SKIPPED", 0) + 1
            continue

        status = process_document(path, record, count_tokens)
        summary[status] = summary.get(status, 0) + 1

    print("Summary:", ", ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
