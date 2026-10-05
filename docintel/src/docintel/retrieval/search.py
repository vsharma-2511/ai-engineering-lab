"""Hybrid search over the latest chunks: embeddings + BM25 keywords.

The two rankings are merged with reciprocal rank fusion (RRF): each
chunk scores sum(1 / (RRF_K + rank)) over the lists it appears in.
Ranks are comparable where raw scores are not (cosine vs. BM25), and a
chunk that is strong in either list rises to the top.

    python -m docintel.retrieval.search "Who prepared this dataset?"
    python -m docintel.retrieval.search "..." --top-k 5 --mode vector
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from .. import config
from ..chunking.chunk_storage import load_latest_chunks
from .bm25 import BM25Index
from .vector_store import load_or_compute_vectors, text_hash

MODES = ("hybrid", "vector", "bm25")

# 60 is the value from the original RRF paper; it damps the gap between
# rank 1 and rank 2 so neither list dominates on its own.
RRF_K = 60


class Retriever:
    def __init__(self, chunks: list[dict], vectors: np.ndarray, encoder):
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(vectors)} vectors for {len(chunks)} chunks"
            )
        self.chunks = chunks
        self.vectors = vectors
        self.encoder = encoder
        self.bm25 = BM25Index([chunk["text"] for chunk in chunks])

    def search(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Best chunks first. Each result is the chunk plus:

        score              ranking score for `mode` (RRF, cosine or BM25)
        vector_similarity  cosine similarity to the question
        bm25_score         keyword score (0 = no shared terms)
        vector_rank, bm25_rank   1-based; bm25_rank is None with no match
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; use one of {MODES}")
        if not question.strip() or not self.chunks:
            return []

        similarity = self.vectors @ self.encoder.encode([question])[0]
        keyword = np.asarray(self.bm25.scores(question))

        vector_order = np.argsort(-similarity, kind="stable")
        vector_rank = np.empty(len(self.chunks), dtype=int)
        vector_rank[vector_order] = np.arange(1, len(self.chunks) + 1)

        bm25_order = [
            p for p in np.argsort(-keyword, kind="stable") if keyword[p] > 0
        ]
        bm25_rank = {int(p): rank for rank, p in enumerate(bm25_order, 1)}

        if mode == "vector":
            score = similarity
        elif mode == "bm25":
            score = keyword
        else:
            score = 1 / (RRF_K + vector_rank)
            for position, rank in bm25_rank.items():
                score[position] += 1 / (RRF_K + rank)

        # Ties (e.g. no keyword matches in bm25 mode) fall back to
        # semantic similarity.
        order = np.lexsort((-similarity, -score))

        return [
            {
                **self.chunks[p],
                "score": float(score[p]),
                "vector_similarity": float(similarity[p]),
                "bm25_score": float(keyword[p]),
                "vector_rank": int(vector_rank[p]),
                "bm25_rank": bm25_rank.get(int(p)),
            }
            for p in order[:top_k]
        ]


def build_retriever(
    db_path: Path = config.DB_PATH,
    model_name: str = config.EMBEDDING_MODEL,
    encoder=None,
) -> Retriever:
    if encoder is None:
        from ..embeddings.encoder import get_encoder
        encoder = get_encoder(model_name)

    chunks = load_latest_chunks(db_path)
    vectors = load_or_compute_vectors(db_path, chunks, encoder)
    return Retriever(chunks, vectors, encoder)


_cache: dict[tuple, tuple[tuple, Retriever]] = {}


def search(
    question: str,
    top_k: int = 5,
    mode: str = "hybrid",
    db_path: Path = config.DB_PATH,
    model_name: str = config.EMBEDDING_MODEL,
) -> list[dict]:
    """Search the latest version of every CHUNKED document.

    The index is kept between calls and rebuilt when the stored chunks
    change (new documents, new versions or reprocessing).
    """
    chunks = load_latest_chunks(db_path)
    signature = tuple(
        (chunk["chunk_id"], text_hash(chunk["text"])) for chunk in chunks
    )
    key = (str(db_path), model_name)

    cached = _cache.get(key)
    if cached is None or cached[0] != signature:
        cached = (signature, build_retriever(db_path, model_name))
        _cache[key] = cached

    return cached[1].search(question, top_k=top_k, mode=mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}")
        return 2

    results = search(args.question, args.top_k, args.mode, args.db)
    if not results:
        print("No chunks found. Run the watcher or reprocess first.")
        return 2

    for number, result in enumerate(results, 1):
        bm25_rank = result["bm25_rank"] or "-"
        print(
            f"{number}. score={result['score']:.4f}  "
            f"cos={result['vector_similarity']:.3f} "
            f"(#{result['vector_rank']})  "
            f"bm25={result['bm25_score']:.2f} (#{bm25_rank})  "
            f"{result['document_name']} p{result['page_number']} "
            f"[{result['content_type']}]"
        )
        print("   " + result["text"].strip().replace("\n", "\n   "))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
