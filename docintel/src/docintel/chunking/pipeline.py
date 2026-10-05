"""Create chunks and prove nothing was lost, duplicated or altered."""
from collections import defaultdict

from .table_chunker import create_table_chunks
from .text_chunker import create_grouped_text_chunks
from ..parsing.cleanup import PAGE_FURNITURE
from ..tokens import TokenCounter

KNOWN_CLASSIFICATIONS = {
    "OUTSIDE_TABLE",
    "INSIDE_TABLE",
    "PARTIAL_OVERLAP",
    PAGE_FURNITURE,
}


def _validate_text_chunks(
    parsed_document: dict,
    text_chunks: list[dict],
    max_tokens: int,
) -> None:
    expected_text = {
        (page["page_number"], block["block_number"]): block["text"]
        for page in parsed_document["pages"]
        for block in page["text_blocks"]
        if block["classification"] == "OUTSIDE_TABLE"
        and block["text"].strip()
    }

    fragments = defaultdict(list)

    for chunk in text_chunks:
        if chunk["token_count"] > max_tokens:
            raise ValueError(
                f"Text chunk {chunk['chunk_id']} has "
                f"{chunk['token_count']} tokens (limit {max_tokens})"
            )

        for span in chunk["source_spans"]:
            start, end = span["start"], span["end"]

            if not 0 <= start < end <= len(chunk["text"]):
                raise ValueError("Invalid text chunk offsets")

            key = (chunk["page_number"], span["block_number"])
            fragments[key].append((
                span["source_start"],
                span["source_end"],
                chunk["text"][start:end],
            ))

    if set(fragments) != set(expected_text):
        raise ValueError("Source text blocks are missing or unexpected")

    for key, original_text in expected_text.items():
        cursor = 0

        for start, end, text in sorted(fragments[key]):
            if start != cursor or not start < end <= len(original_text):
                raise ValueError(f"Missing or overlapping text in {key}")

            if text != original_text[start:end]:
                raise ValueError(f"Text changed in {key}")

            cursor = end

        if cursor != len(original_text):
            raise ValueError(f"Incomplete text coverage in {key}")


def _validate_table_chunks(
    parsed_document: dict,
    table_chunks: list[dict],
) -> None:
    expected_tables = {
        (page["page_number"], table["table_number"]): table["rows"]
        for page in parsed_document["pages"]
        for table in page["tables"]
    }

    parts = defaultdict(list)
    for chunk in table_chunks:
        parts[(chunk["page_number"], chunk["table_number"])].append(chunk)

    if set(parts) != set(expected_tables):
        raise ValueError("Tables are missing or unexpected")

    for key, original_rows in expected_tables.items():
        groups = sorted(parts[key], key=lambda chunk: chunk["part_number"])
        header_row_count = groups[0]["header_row_count"]
        column_names = groups[0]["table"]["column_names"]
        cursor = header_row_count

        for number, chunk in enumerate(groups, start=1):
            if (
                chunk["part_number"] != number
                or chunk["part_count"] != len(groups)
                or chunk["header_row_count"] != header_row_count
                or chunk["table"]["column_names"] != column_names
            ):
                raise ValueError(f"Invalid table group metadata in {key}")

            start = chunk["source_row_start"]
            end = chunk["source_row_end"]

            if start != cursor or not start <= end <= len(original_rows):
                raise ValueError(f"Missing or overlapping table rows in {key}")

            if chunk["table"]["rows"] != original_rows[start:end]:
                raise ValueError(f"Table cells changed in {key}")

            cursor = end

        if cursor != len(original_rows):
            raise ValueError(f"Incomplete table coverage in {key}")


def create_validated_chunks(
    parsed_document: dict,
    count_tokens: TokenCounter,
    max_tokens: int,
    max_rows_per_chunk: int = 25,
) -> list[dict]:
    for page in parsed_document["pages"]:
        for block in page["text_blocks"]:
            if block["classification"] not in KNOWN_CLASSIFICATIONS:
                raise ValueError(
                    f"Unknown classification {block['classification']!r}"
                )

    text_chunks = create_grouped_text_chunks(
        parsed_document,
        count_tokens=count_tokens,
        max_tokens=max_tokens,
    )
    _validate_text_chunks(parsed_document, text_chunks, max_tokens)

    table_chunks = create_table_chunks(
        parsed_document,
        count_tokens=count_tokens,
        max_tokens=max_tokens,
        max_rows_per_chunk=max_rows_per_chunk,
    )
    _validate_table_chunks(parsed_document, table_chunks)

    return text_chunks + table_chunks
