"""Chat sessions with their own documents ("chat and forget").

A file uploaded into a session is processed for that session only: it
is searchable inside the session, never from the library. Ending the
session deletes those documents with everything derived from them.
Sessions idle for longer than SESSION_IDLE_HOURS are ended by `cleanup`,
because people close the browser without ending the chat.

If an upload has the same contents as a library document, the session
reuses the library's chunks instead of processing it again; ending the
session then leaves the library document untouched.

    python -m docintel.sessions new
    python -m docintel.sessions add <session_id> path/to/file.pdf
    python -m docintel.sessions list
    python -m docintel.sessions end <session_id>
    python -m docintel.sessions cleanup [--idle-hours 24]
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

from . import config
from .chunking.chunk_storage import initialize_chunk_storage
from .ingestion.library import delete_document, processing_message
from .ingestion.processor import file_checksum, process_document
from .ingestion.registry import (
    get_latest_document,
    initialize_registry,
    register_document,
    utc_now,
)


def _initialize(db_path: Path) -> None:
    initialize_registry(db_path)
    initialize_chunk_storage(db_path)


def create_session(db_path: Path = config.DB_PATH) -> str:
    _initialize(db_path)
    session_id = uuid.uuid4().hex
    now = utc_now()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)", (session_id, now, now),
        )
    return session_id


def session_exists(session_id: str, db_path: Path = config.DB_PATH) -> bool:
    _initialize(db_path)
    with sqlite3.connect(db_path) as connection:
        return connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,),
        ).fetchone() is not None


def touch_session(session_id: str, db_path: Path = config.DB_PATH) -> None:
    """Record activity, so an active chat is not cleaned up as idle."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            (utc_now(), session_id),
        )


def _link(db_path: Path, session_id: str, document_id: str) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO session_documents VALUES (?, ?)",
            (session_id, document_id),
        )


def _library_copy(db_path: Path, checksum: str) -> dict | None:
    """A searchable library document with exactly these contents."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT d.document_id, d.filename FROM documents AS d
            WHERE d.checksum = ? AND d.session_id IS NULL
              AND d.status = 'CHUNKED'
              AND d.version = (SELECT MAX(version) FROM documents
                               WHERE filename = d.filename)
            """,
            (checksum,),
        ).fetchone()
    return dict(row) if row else None


def add_session_document(
    session_id: str,
    file_path: Path,
    db_path: Path = config.DB_PATH,
    count_tokens=None,
) -> dict:
    """Process an uploaded file for one session. Returns its status.

    status: CHUNKED (searchable), REUSED (identical to a library
    document), NEEDS_REVIEW / FAILED (not searchable; see message),
    DUPLICATE (already uploaded into this session under another name).
    """
    file_path = Path(file_path)
    if not session_exists(session_id, db_path):
        raise ValueError(f"No such session: {session_id}")
    if file_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_path.name}")

    touch_session(session_id, db_path)
    checksum = file_checksum(file_path)

    library = _library_copy(db_path, checksum)
    if library:
        _link(db_path, session_id, library["document_id"])
        return {"status": "REUSED", "filename": file_path.name,
                "document_id": library["document_id"],
                "message": f"Same file as library document "
                           f"{library['filename']}; using its chunks."}

    folder = config.UPLOADS_DIR / session_id
    folder.mkdir(parents=True, exist_ok=True)
    stored = folder / file_path.name
    if file_path.resolve() != stored.resolve():
        shutil.copy2(file_path, stored)

    result, _ = register_document(db_path, stored, checksum, session_id)
    if result == "duplicate_content":
        return {"status": "DUPLICATE", "filename": file_path.name,
                "document_id": None,
                "message": "This session already has a file with the "
                           "same contents."}

    record = get_latest_document(db_path, stored.name, session_id)
    status = record["status"]
    if status not in {"CHUNKED", "NEEDS_REVIEW"}:
        if count_tokens is None:
            from .tokens import get_token_counter
            count_tokens = get_token_counter(config.EMBEDDING_MODEL)
        status = process_document(stored, record, count_tokens, db_path,
                                  config.PARSED_DIR)

    if status == "CHUNKED":
        _link(db_path, session_id, record["document_id"])

    return {"status": status, "filename": stored.name,
            "document_id": record["document_id"],
            "message": processing_message(status, record["document_id"])}


def end_session(session_id: str, db_path: Path = config.DB_PATH) -> dict:
    """Delete the session and every document uploaded into it."""
    _initialize(db_path)
    with sqlite3.connect(db_path) as connection:
        keys = [row[0] for row in connection.execute(
            "SELECT DISTINCT filename FROM documents WHERE session_id = ?",
            (session_id,),
        )]

    deleted = [summary for key in keys
               if (summary := delete_document(db_path, key))]

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "DELETE FROM session_documents WHERE session_id = ?",
            (session_id,),
        )
        connection.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,),
        )

    folder = config.UPLOADS_DIR / session_id
    if folder.is_dir() and folder.resolve().parent == (
        config.UPLOADS_DIR.resolve()
    ):
        shutil.rmtree(folder, ignore_errors=True)

    return {"session_id": session_id, "documents_deleted": deleted}


def list_sessions(db_path: Path = config.DB_PATH) -> list[dict]:
    _initialize(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT s.session_id, s.created_at, s.last_active_at,
                   COUNT(sd.document_id) AS documents
            FROM sessions AS s
            LEFT JOIN session_documents AS sd USING (session_id)
            GROUP BY s.session_id
            ORDER BY s.last_active_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def cleanup_expired_sessions(
    db_path: Path = config.DB_PATH,
    idle_hours: float = config.SESSION_IDLE_HOURS,
) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=idle_hours)
    return [
        end_session(session["session_id"], db_path)
        for session in list_sessions(db_path)
        if datetime.fromisoformat(session["last_active_at"]) < cutoff
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("new", help="Start a session; prints its id")
    add = commands.add_parser("add", help="Upload a PDF into a session")
    add.add_argument("session_id")
    add.add_argument("files", nargs="+", type=Path)
    commands.add_parser("list", help="Show open sessions")
    end = commands.add_parser("end", help="End a session, deleting its files")
    end.add_argument("session_id")
    cleanup = commands.add_parser("cleanup", help="End idle sessions")
    cleanup.add_argument("--idle-hours", type=float,
                         default=config.SESSION_IDLE_HOURS)
    args = parser.parse_args()

    if args.command == "new":
        session_id = create_session()
        print(f"Session: {session_id}\n\nAdd a PDF:\n"
              f"  python -m docintel.sessions add {session_id} file.pdf")
        return 0

    if args.command == "add":
        missing = [path for path in args.files if not path.is_file()]
        if missing and len(args.files) > 1 and Path(
            " ".join(map(str, args.files))
        ).is_file():
            print("That path contains spaces: put it in quotes, e.g.\n"
                  f'  python -m docintel.sessions add {args.session_id} '
                  f'"{" ".join(map(str, args.files))}"')
            return 2
        for path in args.files:
            if not path.is_file():
                print(f"{path}: file not found (current folder: "
                      f"{Path.cwd()})")
                continue
            try:
                result = add_session_document(args.session_id, path)
            except ValueError as exc:
                print(f"{path.name}: {exc}")
                return 2
            print(f"{result['status']}: {result['filename']} | "
                  f"{result['message']}")
        print(f"\nAsk:\n  python -m docintel.qa.ask \"your question\" "
              f"--session {args.session_id}")
        return 0

    if args.command == "list":
        sessions = list_sessions()
        if not sessions:
            print("No open sessions.")
        for session in sessions:
            print(f"{session['session_id']}  documents={session['documents']}"
                  f"  last active {session['last_active_at']}")
        return 0

    if args.command == "end":
        if not session_exists(args.session_id):
            print(f"No such session: {args.session_id}")
            return 2
        ended = [end_session(args.session_id)]
    else:
        ended = cleanup_expired_sessions(idle_hours=args.idle_hours)

    for result in ended:
        print(f"ENDED: {result['session_id']} | documents deleted: "
              f"{len(result['documents_deleted'])}")
    if not ended:
        print("No idle sessions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
