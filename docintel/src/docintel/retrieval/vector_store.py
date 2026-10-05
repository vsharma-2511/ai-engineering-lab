"""Chunk embeddings cached in the registry database.

Vectors are keyed by chunk and model and stored with a hash of the text
they were computed from, so a chunk whose text changes on reprocessing
is re-embedded instead of served a stale vector.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3

import numpy as np


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initialize_vector_storage(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT NOT NULL,
                model TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                dims INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model)
            )
        """)


def load_or_compute_vectors(
    db_path: Path,
    chunks: list[dict],
    encoder,
) -> np.ndarray:
    """One unit-length row per chunk, in the order of `chunks`.

    Only chunks without a current cached vector are sent to the encoder.
    """
    initialize_vector_storage(db_path)

    with sqlite3.connect(db_path) as connection:
        cached = {
            chunk_id: (sha, dims, blob)
            for chunk_id, sha, dims, blob in connection.execute(
                "SELECT chunk_id, text_sha256, dims, vector "
                "FROM chunk_vectors WHERE model = ?",
                (encoder.name,),
            )
        }

    vectors: list[np.ndarray | None] = []
    missing = []

    for position, chunk in enumerate(chunks):
        entry = cached.get(chunk["chunk_id"])
        if entry and entry[0] == text_hash(chunk["text"]):
            vectors.append(np.frombuffer(entry[2], dtype=np.float32))
        else:
            vectors.append(None)
            missing.append(position)

    if missing:
        fresh = encoder.encode([chunks[p]["text"] for p in missing])
        created_at = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(db_path) as connection:
            connection.executemany(
                """
                INSERT INTO chunk_vectors
                    (chunk_id, model, text_sha256, dims, vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, model) DO UPDATE SET
                    text_sha256 = excluded.text_sha256,
                    dims = excluded.dims,
                    vector = excluded.vector,
                    created_at = excluded.created_at
                """,
                [
                    (
                        chunks[p]["chunk_id"],
                        encoder.name,
                        text_hash(chunks[p]["text"]),
                        vector.shape[0],
                        vector.astype(np.float32).tobytes(),
                        created_at,
                    )
                    for p, vector in zip(missing, fresh)
                ],
            )

        for p, vector in zip(missing, fresh):
            vectors[p] = vector

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    return np.vstack(vectors).astype(np.float32)
