"""Token counting, so chunk sizes match what the embedding model sees."""
from functools import lru_cache
import math
import re
from typing import Callable

TokenCounter = Callable[[str], int]

_WORD_OR_SYMBOL = re.compile(r"\w+|[^\w\s]")


@lru_cache(maxsize=4)
def get_token_counter(model_name: str) -> TokenCounter:
    """Exact count using the embedding model's own tokenizer.

    Includes special tokens ([CLS], [SEP]) because they use part of the
    model's input limit too.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return len(
            tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
        )

    return count


def approximate_token_count(text: str) -> int:
    """Tokenizer-free estimate for tests and offline experiments.

    WordPiece splits rare words, so words are weighted by 1.3. This is an
    estimate only; production chunking uses get_token_counter().
    """
    pieces = _WORD_OR_SYMBOL.findall(text)
    return math.ceil(len(pieces) * 1.3) + 2
