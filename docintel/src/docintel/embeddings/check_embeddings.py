"""Health check and retrieval scenarios for the embedding step.

Checks every stored chunk (latest document versions only):
  1. no empty text
  2. fits the model's input limit (no silent truncation)
  3. one finite, unit-length vector per chunk
  4. identical vectors when the same text is encoded twice
Then runs the questions in eval/retrieval_cases.json and reports where
the relevant chunk ranks.

    python -m docintel.embeddings.check_embeddings
    python -m docintel.embeddings.check_embeddings --top-k 5 --show-misses
    python -m docintel.embeddings.check_embeddings --backend tfidf   # offline baseline
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from .. import config
from ..chunking.chunk_storage import load_latest_chunks
from .encoder import SentenceTransformerBackend


class TfidfBackend:
    """Lexical baseline that needs no model download."""

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True
        ).fit(corpus)
        self.name = "tfidf (lexical baseline)"
        self.max_tokens = None

    def count_tokens(self, text: str) -> int:
        return 0

    def encode(self, texts: list[str]) -> np.ndarray:
        # TF-IDF rows are already L2-normalised.
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


def check_vectors(backend, chunks: list[dict]) -> tuple[np.ndarray, list[str]]:
    problems = []

    for chunk in chunks:
        if not chunk["text"].strip():
            problems.append(f"empty text: {chunk['chunk_id']}")

        if backend.max_tokens:
            tokens = backend.count_tokens(chunk["text"])
            if tokens > backend.max_tokens:
                problems.append(
                    f"too long ({tokens} > {backend.max_tokens} tokens, "
                    f"would be truncated): {chunk['chunk_id']}"
                )

    vectors = backend.encode([chunk["text"] for chunk in chunks])

    if vectors.shape[0] != len(chunks):
        problems.append(
            f"{vectors.shape[0]} vectors for {len(chunks)} chunks"
        )
    if not np.isfinite(vectors).all():
        problems.append("vectors contain NaN or infinity")

    norms = np.linalg.norm(vectors, axis=1)
    off = np.flatnonzero(np.abs(norms - 1) > 1e-3)
    for position in off:
        problems.append(
            f"not unit length (norm={norms[position]:.4f}): "
            f"{chunks[position]['chunk_id']}"
        )

    sample = [chunk["text"] for chunk in chunks[:8]]
    if sample and not np.allclose(
        backend.encode(sample), vectors[:len(sample)], atol=1e-5
    ):
        problems.append("re-encoding the same text gave different vectors")

    seen = defaultdict(list)
    for chunk in chunks:
        seen[chunk["text"].strip()].append(chunk["chunk_id"])
    for ids in seen.values():
        if len(ids) > 1:
            problems.append(f"duplicate chunk text: {', '.join(ids)}")

    return vectors, problems


def is_relevant(text: str, expect_any: list[list[str]]) -> bool:
    return any(all(phrase in text for phrase in group) for group in expect_any)


def short_id(chunk: dict) -> str:
    return chunk["chunk_id"].split(":", 1)[-1]


def run_cases(backend, chunks, vectors, cases, top_k, show_misses):
    results = []

    for case in cases:
        question_vector = backend.encode([case["question"]])[0]
        scores = vectors @ question_vector
        order = np.argsort(-scores)

        relevant_positions = [
            position for position in order
            if is_relevant(chunks[position]["text"], case["expect_any"])
        ]
        rank = (
            int(np.flatnonzero(order == relevant_positions[0])[0]) + 1
            if relevant_positions else None
        )

        results.append({
            **case,
            "rank": rank,
            "top_score": float(scores[order[0]]),
            "relevant_score": (
                float(scores[relevant_positions[0]])
                if relevant_positions else None
            ),
            "top_chunks": [
                (short_id(chunks[p]), float(scores[p]), chunks[p]["text"])
                for p in order[:top_k]
            ],
            "has_relevant_chunk": bool(relevant_positions),
        })

    answerable = [r for r in results if r["expect_any"]]
    unanswerable = [r for r in results if not r["expect_any"]]

    print(f"\n{'case':32} {'category':24} {'rank':>5} {'top':>6} {'rel':>6}")
    print("-" * 78)

    for r in results:
        if not r["expect_any"]:
            verdict = "n/a"
        elif not r["has_relevant_chunk"]:
            verdict = "NOCHUNK"
        elif r["rank"] == 1:
            verdict = "1"
        elif r["rank"] <= top_k:
            verdict = str(r["rank"])
        else:
            verdict = f"MISS({r['rank']})"

        relevant = (
            f"{r['relevant_score']:.3f}" if r["relevant_score"] is not None
            else "-"
        )
        print(
            f"{r['id']:32} {r['category']:24} {verdict:>5} "
            f"{r['top_score']:6.3f} {relevant:>6}"
        )

    no_chunk = [r for r in answerable if not r["has_relevant_chunk"]]
    hits_1 = sum(r["rank"] == 1 for r in answerable)
    hits_k = sum(r["rank"] is not None and r["rank"] <= top_k
                 for r in answerable)
    mrr = sum(1 / r["rank"] for r in answerable if r["rank"]) / max(
        len(answerable), 1
    )

    print("-" * 78)
    print(
        f"Answerable cases: {len(answerable)} | hit@1 {hits_1}/"
        f"{len(answerable)} | hit@{top_k} {hits_k}/{len(answerable)} | "
        f"MRR {mrr:.3f}"
    )

    by_category = defaultdict(list)
    for r in answerable:
        by_category[r["category"]].append(r)
    for category, rows in sorted(by_category.items()):
        ok = sum(r["rank"] is not None and r["rank"] <= top_k for r in rows)
        print(f"  {category:26} hit@{top_k} {ok}/{len(rows)}")

    if no_chunk:
        print(
            "\nNo chunk contains the expected text for: "
            + ", ".join(r["id"] for r in no_chunk)
            + "\n  -> a parsing/chunking problem, not an embedding one."
        )

    if unanswerable:
        worst_unanswerable = max(r["top_score"] for r in unanswerable)
        answered_top1 = [r["top_score"] for r in answerable if r["rank"] == 1]
        weakest_hit = min(answered_top1) if answered_top1 else None
        print(
            f"\nHighest score on an unanswerable question: "
            f"{worst_unanswerable:.3f}"
        )
        if weakest_hit is not None:
            print(f"Lowest top score on a correct answer:     {weakest_hit:.3f}")
            if worst_unanswerable < weakest_hit:
                print(
                    "  -> a similarity threshold between these can flag "
                    "'not in the documents'."
                )
            else:
                print(
                    "  -> scores overlap: a threshold alone cannot detect "
                    "unanswerable questions (the QA prompt must handle it)."
                )

    if show_misses:
        for r in results:
            if r["expect_any"] and r["rank"] == 1:
                continue
            print(f"\n=== {r['id']}: {r['question']}")
            for chunk_id, score, text in r["top_chunks"]:
                preview = text.replace("\n", " / ")[:160]
                print(f"  {score:.3f}  {chunk_id}\n         {preview}")

    return hits_k, len(answerable), len(no_chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["model", "tfidf"], default="model",
    )
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument(
        "--cases", type=Path,
        default=config.EVAL_DIR / "retrieval_cases.json",
    )
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--show-misses", action="store_true",
        help="Print the top chunks for every case not ranked first",
    )
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}")
        return 2

    chunks = load_latest_chunks(args.db)
    if not chunks:
        print("No chunks found. Run the watcher or reprocess first.")
        return 2

    documents = {(c["document_name"], c["document_version"]) for c in chunks}
    print(f"Chunks: {len(chunks)} from {len(documents)} document(s)")
    for name, version in sorted(documents):
        print(f"  v{version}  {name}")

    if args.backend == "tfidf":
        backend = TfidfBackend([chunk["text"] for chunk in chunks])
    else:
        backend = SentenceTransformerBackend(args.model)

    print(f"Backend: {backend.name}"
          + (f" (input limit {backend.max_tokens} tokens)"
             if backend.max_tokens else ""))

    vectors, problems = check_vectors(backend, chunks)

    print(f"\nVector checks: {vectors.shape[0]} vectors x "
          f"{vectors.shape[1]} dims")
    if problems:
        for problem in problems:
            print(f"  PROBLEM: {problem}")
    else:
        print("  all passed")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    hits, total, no_chunk = run_cases(
        backend, chunks, vectors, cases, args.top_k, args.show_misses,
    )

    failed = bool(problems) or no_chunk or hits < total
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
