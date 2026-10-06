"""Run eval/retrieval_cases.json through the full QA step.

A case passes when:
  answerable    status is "answered" and a verified citation points at
                a chunk containing the expected text
  unanswerable  status is "not_found"

Each case is one LLM call, so this costs money (or free-tier quota).

    python -m docintel.qa.evaluate
    python -m docintel.qa.evaluate --provider claude --sleep 4
    python -m docintel.qa.evaluate --only table_total_2024,text_trend

The run stops at the first rate-limit/quota error and prints a command
that re-runs only the cases left unfinished.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from .. import config
from ..embeddings.check_embeddings import is_relevant
from ..retrieval.search import build_retriever
from .answer import answer_from_chunks
from .llm import PROVIDERS, LLMError, LLMQuotaError, get_llm


def judge(case: dict, result: dict, chunks_by_id: dict) -> str:
    if not case["expect_any"]:
        return "PASS" if result["status"] == "not_found" else "FAIL"
    if result["status"] != "answered":
        return "FAIL"
    cited = [chunks_by_id[c["chunk_id"]] for c in result["citations"]]
    return (
        "PASS" if any(is_relevant(chunk["text"], case["expect_any"])
                      for chunk in cited)
        else "FAIL"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument(
        "--cases", type=Path,
        default=config.EVAL_DIR / "retrieval_cases.json",
    )
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--top-k", type=int, default=config.QA_TOP_K)
    parser.add_argument("--sleep", type=float, default=0,
                        help="Seconds between questions (free-tier limits)")
    parser.add_argument("--only",
                        help="Comma-separated case ids to run")
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}")
        return 2

    try:
        llm = get_llm(args.provider, args.model)
    except LLMError as exc:
        print(f"Error: {exc}")
        return 2

    retriever = build_retriever(args.db)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in retriever.chunks}

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            print(f"Unknown case ids: {', '.join(sorted(unknown))}")
            return 2
        cases = [case for case in cases if case["id"] in wanted]

    print(f"QA eval: {llm.provider} / {llm.model} | {len(cases)} cases | "
          f"top_k={args.top_k}\n")

    verdicts = []
    quota_error = None
    for number, case in enumerate(cases):
        if number and args.sleep:
            time.sleep(args.sleep)

        chunks = retriever.search(case["question"], top_k=args.top_k)
        try:
            result = answer_from_chunks(case["question"], chunks, llm)
            verdict = judge(case, result, chunks_by_id)
        except LLMQuotaError as exc:
            # Every remaining case would fail the same way.
            quota_error = exc
            break
        except Exception as exc:  # report and keep going
            result = {"status": "error", "answer": str(exc),
                      "citations": [], "rejected_citations": []}
            verdict = "ERROR"
        verdicts.append(verdict)

        pages = ",".join(f"p{c['page']}" for c in result["citations"]) or "-"
        print(f"{verdict:5} {case['id']:30} {result['status']:11} "
              f"cites={pages}")
        if verdict != "PASS":
            print(f"      Q: {case['question']}")
            print(f"      A: {result['answer'][:200] or '(none)'}")
            for rejected in result["rejected_citations"]:
                print(f"      rejected [{rejected.get('source')}] "
                      f"{rejected['reason']}: {rejected.get('quote', '')[:80]}")

    passed = verdicts.count("PASS")
    print(f"\nPassed {passed}/{len(verdicts)} run"
          + (f" | errors {verdicts.count('ERROR')}"
             if "ERROR" in verdicts else ""))

    if quota_error:
        print(f"\nStopped after {len(verdicts)} of {len(cases)} cases: "
              f"{quota_error}")

    redo = [
        case["id"] for case, verdict in zip(cases, verdicts)
        if verdict == "ERROR"
    ] + [case["id"] for case in cases[len(verdicts):]]
    if redo:
        print("\nRe-run the errored and unfinished cases with:\n"
              f"  python -m docintel.qa.evaluate --only {','.join(redo)}"
              f" --provider {llm.provider} --model {llm.model}")

    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
