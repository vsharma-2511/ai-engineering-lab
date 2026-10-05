"""Okapi BM25 keyword scoring.

Embeddings miss exact-word matches that a keyword index finds easily,
e.g. "Who prepared this dataset?" against a "Source: ... prepared ..."
line. Small enough to keep in memory: one entry per chunk.
"""
from collections import Counter
import math
import re

_WORD = re.compile(r"\w+")

# Question words carry no evidence about which chunk answers.
STOPWORDS = frozenset("""
a an and are as at be by did do does for from had has have how in is it
its of on or that the their there this to was were what when where which
who whom why will with
""".split())


def tokenize(text: str) -> list[str]:
    return [
        word for word in _WORD.findall(text.lower())
        if word not in STOPWORDS
    ]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0
        )

        document_frequency = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())

        total = len(texts)
        self.idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        results = []

        for counts, length in zip(self.term_counts, self.lengths):
            norm = self.k1 * (
                1 - self.b + self.b * length / (self.average_length or 1)
            )
            score = 0.0
            for term in terms:
                frequency = counts.get(term)
                if frequency:
                    score += self.idf[term] * (
                        frequency * (self.k1 + 1) / (frequency + norm)
                    )
            results.append(score)

        return results
