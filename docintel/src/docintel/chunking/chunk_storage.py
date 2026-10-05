import sqlite3
from pathlib import Path

import json
from datetime import datetime, timezone


def save_chunks(db_path: Path, chunks: list[dict]) -> None:
    if not chunks:
        raise ValueError("No chunks to save")

    document_versions = {
        (chunk["document_id"], chunk["document_version"])
        for chunk in chunks
    }

    if len(document_versions) != 1:
        raise ValueError("Save one document version at a time")

    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")

    document_id, document_version = next(iter(document_versions))


    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        connection.execute("BEGIN IMMEDIATE")

        connection.execute(
            """
            DELETE FROM chunks
            WHERE document_id = ? AND document_version = ?
            """,
            (document_id, document_version),
        )

        for chunk in chunks:
            # Everything beyond the common columns becomes metadata.
            metadata = {
                key: value
                for key, value in chunk.items()
                if key not in {
                    "chunk_id",
                    "document_id",
                    "document_version",
                    "document_name",
                    "page_number",
                    "content_type",
                    "text",
                }
            }

            connection.execute(
                """
                INSERT INTO chunks (
                    chunk_id,
                    document_id,
                    document_version,
                    document_name,
                    page_number,
                    content_type,
                    text,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    text = excluded.text,
                    metadata_json = excluded.metadata_json
                """,
                (
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk["document_version"],
                    chunk["document_name"],
                    chunk["page_number"],
                    chunk["content_type"],
                    chunk["text"],
                    json.dumps(metadata, ensure_ascii=False),
                    created_at,
                ),
            )

def initialize_chunk_storage(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_version INTEGER NOT NULL,
                document_name TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_version
            ON chunks(document_id, document_version)
        """)

def load_latest_chunks(db_path: Path) -> list[dict]:
    """Chunks of the newest CHUNKED version of each document.

    Older versions stay in the table for history but must never be
    retrieved, or answers could quote superseded documents.
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT c.chunk_id, c.document_id, c.document_version,
                   c.document_name, c.page_number, c.content_type,
                   c.text, c.metadata_json
            FROM chunks AS c
            JOIN documents AS d
              ON d.document_id = c.document_id
             AND d.version = c.document_version
            WHERE d.status = 'CHUNKED'
              AND d.version = (
                  SELECT MAX(version) FROM documents
                  WHERE filename = d.filename
              )
            ORDER BY c.document_name, c.page_number, c.chunk_id
            """
        ).fetchall()

    chunks = []
    for row in rows:
        chunk = dict(row)
        chunk.update(json.loads(chunk.pop("metadata_json")))
        chunks.append(chunk)

    return chunks
