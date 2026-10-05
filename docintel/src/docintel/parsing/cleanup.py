"""Detect repeated page headers/footers ("page furniture").

Footers such as "Company confidential - Page 3" add the same noise to
every page's chunks and pull unrelated chunks together in embedding
space. They are marked PAGE_FURNITURE and left out of text chunks.
"""
from collections import Counter
import math
import re

PAGE_FURNITURE = "PAGE_FURNITURE"

PAGE_NUMBER_ONLY = re.compile(
    r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    # Page numbers differ per page, so digits are masked before comparing.
    text = re.sub(r"\d+", "#", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _edge_blocks(page: dict, edge_count: int) -> list[dict]:
    blocks = sorted(
        (
            block for block in page["text_blocks"]
            if block["classification"] == "OUTSIDE_TABLE"
            and block["text"].strip()
        ),
        key=lambda block: (block["bbox"][1], block["bbox"][0]),
    )
    edges = blocks[:edge_count] + blocks[-edge_count:]

    unique = {}
    for block in edges:
        unique[block["block_number"]] = block

    return list(unique.values())


def mark_page_furniture(
    pages: list[dict],
    edge_count: int = 2,
    max_characters: int = 200,
) -> list[dict]:
    """Mark header/footer blocks in place and return the marked blocks.

    A block is furniture when it sits among the first or last blocks of
    its page, is short, and either is only a page number or repeats
    (ignoring digits) on at least half the pages, with a minimum of two.
    Single-page documents only lose bare page numbers.
    """
    edges_by_page = {
        page["page_number"]: _edge_blocks(page, edge_count)
        for page in pages
    }

    occurrences = Counter()
    for blocks in edges_by_page.values():
        occurrences.update({_normalize(block["text"]) for block in blocks})

    threshold = max(2, math.ceil(len(pages) / 2))
    marked = []

    for blocks in edges_by_page.values():
        for block in blocks:
            text = block["text"].strip()

            if len(text) > max_characters:
                continue

            if (
                PAGE_NUMBER_ONLY.match(text)
                or occurrences[_normalize(text)] >= threshold
            ):
                block["classification"] = PAGE_FURNITURE
                block["table_number"] = None
                marked.append(block)

    return marked
