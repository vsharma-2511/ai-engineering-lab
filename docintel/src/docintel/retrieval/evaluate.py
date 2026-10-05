"""Compare retrieval modes on eval/retrieval_cases.json.

Shows the rank of the first relevant chunk per case for vector-only,
BM25-only and hybrid search, then hit@1, hit@k and MRR per mode.

    python -m docintel.retrieval.evaluate
    python -m docintel.retrieval.evaluate --top-k 5
"""
import argparse
import json
from pathlib import Path
import sys

from .. import config
from ..embeddings.check_embeddings import is_relevant
from .search import MODES, build_retriever


def first_relevant_rank(results: list[dict], expect_any) -> int | None:
    for rank, result in enumerate(results, 1):
        if is_relevant(result["text"], expect_any):
            return rank
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path,
        default=config.EVAL_DIR / "retrieval_cases.json",
    )
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}")
        return 2

    retriever = build_retriever(args.db)
    if not retriever.chunks:
        print("No chunks found. Run the watcher or reprocess first.")
        return 2

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    answerable = [case for case in cases if case["expect_any"]]
    total = len(retriever.chunks)

    print(f"Chunks: {total} | answerable cases: {len(answerable)}\n")
    print(f"{'case':32} " + " ".join(f"{mode:>8}" for mode in MODES))
    print("-" * (33 + 9 * len(MODES)))

    ranks = {mode: [] for mode in MODES}

    for case in answerable:
        row = []
        for mode in MODES:
            # Rank over every chunk so misses show how far off they are.
            results = retriever.search(case["question"], top_k=total,
                                       mode=mode)
            rank = first_relevant_rank(results, case["expect_any"])
            ranks[mode].append(rank)
            label = "-" if rank is None else str(rank)
            if rank is not None and rank > args.top_k:
                label = f"MISS({rank})"
            row.append(f"{label:>8}")
        print(f"{case['id']:32} " + " ".join(row))

    print("-" * (33 + 9 * len(MODES)))

    hits_by_mode = {}
    for mode in MODES:
        found = [r for r in ranks[mode] if r is not None]
        hits_1 = sum(r == 1 for r in found)
        hits_k = sum(r <= args.top_k for r in found)
        mrr = sum(1 / r for r in found) / len(answerable)
        hits_by_mode[mode] = hits_k
        print(
            f"{mode:8} hit@1 {hits_1}/{len(answerable)} | "
            f"hit@{args.top_k} {hits_k}/{len(answerable)} | MRR {mrr:.3f}"
        )

    return 0 if hits_by_mode["hybrid"] == len(answerable) else 1


if __name__ == "__main__":
    sys.exit(main())
