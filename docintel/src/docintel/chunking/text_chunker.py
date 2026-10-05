"""Group text blocks outside tables into token-limited chunks.

Every chunk records source_spans, so each character can be traced back
to its original block (used for validation and, later, citations).
"""
import re

from ..parsing.cleanup import PAGE_FURNITURE
from ..tokens import TokenCounter

SENTENCE_END = re.compile(r"[.!?](?:\s+|$)")
WHITESPACE_END = re.compile(r"\s+")


def is_section_heading(text: str) -> bool:
    lines = text.strip().splitlines()

    return (
        len(lines) == 1
        and re.match(r"^\d+\.\s+\S", lines[0]) is not None
    )


def _largest_fitting_boundary(
    text: str,
    start: int,
    boundaries: list[int],
    max_tokens: int,
    count_tokens: TokenCounter,
) -> int | None:
    """Binary search for the last boundary whose span still fits."""
    candidates = [position for position in boundaries if position > start]
    low, high = 0, len(candidates) - 1
    best = None

    while low <= high:
        middle = (low + high) // 2
        position = candidates[middle]

        if count_tokens(text[start:position]) <= max_tokens:
            best = position
            low = middle + 1
        else:
            high = middle - 1

    return best


def _hard_cut(
    text: str,
    start: int,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> int:
    """Largest end position that fits, for text without usable boundaries."""
    low, high = start + 1, len(text)
    best = start + 1

    while low <= high:
        middle = (low + high) // 2

        if count_tokens(text[start:middle]) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    return best


def split_text_block(
    text: str,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[dict]:
    """Split one block into fragments of at most max_tokens.

    Prefers sentence ends, then whitespace, then a hard cut. Fragments
    cover the block exactly, without gaps or overlap.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    sentence_ends = [match.end() for match in SENTENCE_END.finditer(text)]
    whitespace_ends = [match.end() for match in WHITESPACE_END.finditer(text)]

    fragments = []
    start = 0

    while start < len(text):
        if count_tokens(text[start:]) <= max_tokens:
            end = len(text)
        else:
            end = (
                _largest_fitting_boundary(
                    text, start, sentence_ends, max_tokens, count_tokens
                )
                or _largest_fitting_boundary(
                    text, start, whitespace_ends, max_tokens, count_tokens
                )
                or _hard_cut(text, start, max_tokens, count_tokens)
            )

        fragments.append({
            "text": text[start:end],
            "source_start": start,
            "source_end": end,
        })
        start = end

    return fragments


def create_grouped_text_chunks(
    parsed_document: dict,
    count_tokens: TokenCounter,
    max_tokens: int,
) -> list[dict]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    chunks = []
    current_section = None

    for page in parsed_document["pages"]:
        current_text = ""
        source_spans = []

        def flush():
            nonlocal current_text, source_spans

            if not source_spans:
                return

            chunks.append({
                "chunk_id": (
                    f"{parsed_document['document_id']}"
                    f":v{parsed_document['document_version']}"
                    f":p{page['page_number']}"
                    f":g{len(chunks) + 1}"
                ),
                "document_id": parsed_document["document_id"],
                "document_name": parsed_document["document_name"],
                "document_version": parsed_document["document_version"],
                "page_number": page["page_number"],
                "content_type": "text",
                "text": current_text,
                "token_count": count_tokens(current_text),
                "source_spans": source_spans,
                "section": current_section,
            })

            current_text = ""
            source_spans = []

        for block in page["text_blocks"]:
            classification = block["classification"]

            if classification == "PARTIAL_OVERLAP":
                raise ValueError(
                    f"Review overlapping block {block['block_number']} "
                    f"on page {page['page_number']}"
                )

            if classification == PAGE_FURNITURE:
                continue

            if classification == "INSIDE_TABLE":
                flush()
                continue

            if classification != "OUTSIDE_TABLE":
                raise ValueError(f"Unknown classification: {classification}")

            if not block["text"].strip():
                continue

            if is_section_heading(block["text"]):
                # Save previous text under its previous section.
                flush()
                current_section = block["text"].strip()

            fragments = split_text_block(
                block["text"], max_tokens, count_tokens
            )

            for fragment in fragments:
                # Separate different blocks, but don't insert characters
                # between fragments of the same block.
                same_block = (
                    source_spans
                    and source_spans[-1]["block_number"]
                    == block["block_number"]
                )
                separator = "\n" if source_spans and not same_block else ""

                if source_spans and count_tokens(
                    current_text + separator + fragment["text"]
                ) > max_tokens:
                    flush()
                    separator = ""

                start = len(current_text) + len(separator)
                current_text += separator + fragment["text"]

                source_spans.append({
                    "block_number": block["block_number"],
                    "source_start": fragment["source_start"],
                    "source_end": fragment["source_end"],
                    "start": start,
                    "end": len(current_text),
                })

        flush()  # Save the final group on this page.

    return chunks
