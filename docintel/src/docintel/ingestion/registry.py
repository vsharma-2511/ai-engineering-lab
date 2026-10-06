from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_key(filename: str, session_id: str | None = None) -> str:
    """The `filename` column: versions are counted per key.

    Library documents use their filename. A session's uploads get their
    own namespace, so uploading "Report.pdf" into a chat never becomes a
    new version of the library's "Report.pdf".
    """
    return filename if session_id is None else f"session:{session_id}/{filename}"


def initialize_registry(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                checksum TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                source_path TEXT NOT NULL,
                archive_path TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (filename, version)
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_checksum
            ON documents (checksum)
        """)

        # Added for chat sessions. NULL means a library document.
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(documents)"
        )}
        if "session_id" not in columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN session_id TEXT"
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL
            )
        """)

        # Every document a session can search: its own uploads plus
        # library documents it reused instead of processing again.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS session_documents (
                session_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                PRIMARY KEY (session_id, document_id)
            )
        """)


def register_document(
    db_path: Path,
    path: Path,
    checksum: str,
    session_id: str | None = None,
) -> tuple[str, int | None]:
    """
    Register a stable file in the library, or in a chat session.

    Returns:
        ("already_registered", version) for the same filename and contents
        ("duplicate_content", None) if another filename in the same scope
            (library or this session) has these contents
        ("registered", version) for a new document or version
    """
    key = registry_key(path.name, session_id)

    with sqlite3.connect(db_path) as connection:
        # Serialize the check-and-insert operation.
        connection.execute("BEGIN IMMEDIATE")

        latest = connection.execute(
            """
            SELECT version, checksum
            FROM documents
            WHERE filename = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (key,),
        ).fetchone()

        if latest is not None and latest[1] == checksum:
            return "already_registered", latest[0]

        duplicate = connection.execute(
            """
            SELECT document_id
            FROM documents
            WHERE checksum = ? AND session_id IS ?
            LIMIT 1
            """,
            (checksum, session_id),
        ).fetchone()

        if duplicate is not None:
            return "duplicate_content", None

        version = 1 if latest is None else latest[0] + 1
        now = utc_now()

        connection.execute(
            """
            INSERT INTO documents (
                document_id, filename, checksum, version, status,
                source_path, session_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                key,
                checksum,
                version,
                "DISCOVERED",
                str(path),
                session_id,
                now,
                now,
            ),
        )

        return "registered", version

DOCUMENT_COLUMNS = """
    document_id, filename, checksum, version, status, source_path,
    archive_path, session_id, error_message
"""


def get_latest_document(
    db_path: Path,
    filename: str,
    session_id: str | None = None,
) -> dict | None:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            f"""
            SELECT {DOCUMENT_COLUMNS}
            FROM documents
            WHERE filename = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (registry_key(filename, session_id),),
        ).fetchone()

    return dict(row) if row else None


def update_document_status(
    db_path: Path,
    document_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE documents
            SET status = ?, error_message = ?, updated_at = ?
            WHERE document_id = ?
            """,
            (status, error_message, utc_now(), document_id),
        )

def get_document_versions(db_path: Path, key: str) -> list[dict]:
    """Every registered version of one registry key, oldest first."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT {DOCUMENT_COLUMNS}
            FROM documents
            WHERE filename = ?
            ORDER BY version
            """,
            (key,),
        ).fetchall()

    return [dict(row) for row in rows]


def set_archive_path(db_path: Path, document_id: str, path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE documents SET archive_path = ?, updated_at = ?
            WHERE document_id = ?
            """,
            (str(path), utc_now(), document_id),
        )


def get_latest_documents(db_path: Path) -> list[dict]:
    """The newest registered version of every registry key.

    Includes chat-session uploads; their `session_id` is set.
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT d.document_id, d.filename, d.checksum, d.version,
                   d.status, d.source_path, d.archive_path, d.session_id,
                   d.error_message
            FROM documents AS d
            JOIN (
                SELECT filename, MAX(version) AS version
                FROM documents
                GROUP BY filename
            ) AS latest
              ON latest.filename = d.filename
             AND latest.version = d.version
            ORDER BY d.filename
            """
        ).fetchall()

    return [dict(row) for row in rows]
