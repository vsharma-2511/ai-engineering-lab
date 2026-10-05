"""Sentence-transformers wrapper shared by the embedding check and retrieval."""
from functools import lru_cache

import numpy as np


class SentenceTransformerBackend:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.max_tokens = self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(
            text, add_special_tokens=True, truncation=False, verbose=False,
        )["input_ids"])

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-length float32 rows, so a dot product is cosine similarity."""
        vectors = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)


@lru_cache(maxsize=2)
def get_encoder(model_name: str) -> SentenceTransformerBackend:
    """Load the model once per process."""
    return SentenceTransformerBackend(model_name)
