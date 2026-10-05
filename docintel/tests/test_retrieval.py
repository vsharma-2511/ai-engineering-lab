"""Retrieval tests with a deterministic fake encoder (no model download).

Run from the project folder:  pytest
"""
import numpy as np
import pytest

from docintel.retrieval.bm25 import BM25Index, tokenize
from docintel.retrieval.search import Retriever
from docintel.retrieval.vector_store import load_or_compute_vectors


class FakeEncoder:
    """Bag of words hashed into 64 dims, unit length. Counts its calls."""

    name = "fake-encoder"

    def __init__(self):
        self.encoded = []

    def encode(self, texts):
        self.encoded.extend(texts)
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in tokenize(text):
                vectors[row, sum(map(ord, word)) % 64] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


def make_chunks(texts):
    return [
        {"chunk_id": f"doc:v1:g{i}", "text": text, "document_name": "t.pdf",
         "page_number": 1, "content_type": "text"}
        for i, text in enumerate(texts)
    ]


CHUNKS = make_chunks([
    "Table 1. Library visits / Year: 2023 | Visits: 412",
    "Source: Fictional Northport Service Office, prepared September 2026.",
    "Completion rate is the share of participants who finished a program.",
])


def test_tokenize_drops_question_words_and_lowercases():
    assert tokenize("Who prepared THIS dataset?") == ["prepared", "dataset"]


def test_bm25_scores_only_matching_chunks():
    scores = BM25Index([c["text"] for c in CHUNKS]).scores(
        "who prepared it"
    )
    assert scores[1] > 0
    assert scores[0] == scores[2] == 0


def test_bm25_rarer_term_scores_higher():
    index = BM25Index(["apple banana", "apple cherry", "apple durian"])
    common, rare = index.scores("apple")[0], index.scores("banana")[0]
    assert rare > common


@pytest.fixture
def retriever():
    encoder = FakeEncoder()
    return Retriever(CHUNKS, encoder.encode([c["text"] for c in CHUNKS]),
                     encoder)


# Vector mode is left out: the hashed fake encoder has no semantics.
@pytest.mark.parametrize("mode", ["hybrid", "bm25"])
def test_search_finds_keyword_chunk(retriever, mode):
    results = retriever.search("Who prepared the dataset?", top_k=2,
                               mode=mode)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "doc:v1:g1"


def test_search_result_fields(retriever):
    top = retriever.search("library visits 2023", top_k=3)
    assert [r["chunk_id"] for r in top][0] == "doc:v1:g0"
    assert top[0]["text"] == CHUNKS[0]["text"]
    assert top[0]["vector_rank"] == 1 and top[0]["bm25_rank"] == 1
    assert sorted(r["vector_rank"] for r in top) == [1, 2, 3]
    # Chunks sharing no words with the question have no keyword rank.
    assert any(r["bm25_rank"] is None for r in top)
    assert [r["score"] for r in top] == sorted(
        (r["score"] for r in top), reverse=True
    )


def test_hybrid_rescues_chunk_missed_by_vectors():
    # Vectors rank chunk 1 last; BM25 ranks it first on an exact term.
    chunks = make_chunks(["alpha beta", "gamma zeta", "alpha delta"])
    vectors = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)

    class QuestionEncoder:
        name = "q"
        def encode(self, texts):
            return np.array([[1, 0]], dtype=np.float32)

    retriever = Retriever(chunks, vectors, QuestionEncoder())
    vector_only = retriever.search("zeta", top_k=3, mode="vector")
    hybrid = retriever.search("zeta", top_k=3, mode="hybrid")

    assert vector_only[-1]["chunk_id"] == "doc:v1:g1"
    # g1: 1/(60+3) + 1/(60+1) beats g0: 1/(60+1) with no keyword match.
    assert hybrid[0]["chunk_id"] == "doc:v1:g1"
    assert hybrid[0]["vector_rank"] == 3 and hybrid[0]["bm25_rank"] == 1


def test_empty_question_and_unknown_mode(retriever):
    assert retriever.search("   ") == []
    with pytest.raises(ValueError):
        retriever.search("visits", mode="fuzzy")


def test_vectors_are_cached_and_refreshed_on_text_change(tmp_path):
    db = tmp_path / "registry.db"
    encoder = FakeEncoder()

    first = load_or_compute_vectors(db, CHUNKS, encoder)
    assert first.shape == (3, 64)
    assert len(encoder.encoded) == 3

    again = load_or_compute_vectors(db, CHUNKS, encoder)
    assert len(encoder.encoded) == 3  # nothing re-encoded
    assert np.array_equal(first, again)

    changed = [dict(CHUNKS[0], text="Rewritten table text"), *CHUNKS[1:]]
    load_or_compute_vectors(db, changed, encoder)
    assert encoder.encoded[3:] == ["Rewritten table text"]
