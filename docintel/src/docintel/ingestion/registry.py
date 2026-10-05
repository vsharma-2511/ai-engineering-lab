from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def register_document(
    db_path: Path,
    path: Path,
    checksum: str,
) -> tuple[str, int | None]:
    """
    Register a stable file.

    Returns:
        ("already_registered", version) for the same filename and contents
        ("duplicate_content", None) if another filename has these contents
        ("registered", version) for a new document or version
    """
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
            (path.name,),
        ).fetchone()

        if latest is not None and latest[1] == checksum:
            return "already_registered", latest[0]

        duplicate = connection.execute(
            """
            SELECT document_id
            FROM documents
            WHERE checksum = ?
            LIMIT 1
            """,
            (checksum,),
        ).fetchone()

        if duplicate is not None:
            return "duplicate_content", None

        version = 1 if latest is None else latest[0] + 1
        now = utc_now()

        connection.execute(
            """
            INSERT INTO documents (
                document_id, filename, checksum, version, status,
                source_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                path.name,
                checksum,
                version,
                "DISCOVERED",
                str(path),
                now,
                now,
            ),
        )

        return "registered", version

def get_latest_document(db_path: Path, filename: str) -> dict | None:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT document_id, filename, checksum, version, status
            FROM documents
            WHERE filename = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (filename,),
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

def get_latest_documents(db_path: Path) -> list[dict]:
    """The newest registered version of every filename."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT d.document_id, d.filename, d.checksum, d.version,
                   d.status, d.source_path, d.error_message
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
