"""Ask a question about the processed documents.

    python -m docintel.qa.ask "How many people joined Digital skills at East?"
    python -m docintel.qa.ask "..." --provider openai --model gpt-5.4-mini
    python -m docintel.qa.ask "..." --show-sources
"""
import argparse
from pathlib import Path
import sys

from .. import config
from .answer import answer
from .llm import PROVIDERS, LLMError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        help=f"default: {config.QA_PROVIDER}")
    parser.add_argument("--model", help="default: the provider's default")
    parser.add_argument("--top-k", type=int, default=config.QA_TOP_K)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--show-sources", action="store_true",
                        help="List every retrieved chunk, cited or not")
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}")
        return 2

    try:
        result = answer(args.question, args.top_k, args.provider,
                        args.model, args.db)
    except LLMError as exc:
        print(f"Error: {exc}")
        return 2

    print(f"[{result['provider']} / {result['model']}]  "
          f"status: {result['status']}\n")

    if result["status"] == "not_found":
        print("The documents do not contain an answer to this question.")
    else:
        if result["status"] == "unsupported":
            print("WARNING: none of the quotes were found in the sources; "
                  "do not rely on this answer.\n")
        print(result["answer"])

    if result["citations"]:
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"  [{citation['source']}] {citation['document']}, "
                  f"page {citation['page']}: \"{citation['quote']}\"")

    for rejected in result["rejected_citations"]:
        print(f"  rejected [{rejected.get('source')}]: {rejected['reason']}")

    if args.show_sources:
        print("\nRetrieved sources:")
        for source in result["sources"]:
            print(f"  [{source['source']}] {source['document']}, "
                  f"page {source['page']}  ({source['chunk_id']})")

    return 0 if result["status"] != "unsupported" else 1


if __name__ == "__main__":
    sys.exit(main())
