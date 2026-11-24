"""Second-stage rerankers. Each returns scores in [0, 1], so the abstention gate
can use a single threshold whichever reranker is active."""

from __future__ import annotations

import math
from typing import Protocol

from askdocs.config import RerankConfig
from askdocs.retrieval.bm25 import BM25Index
from askdocs.text import tokenize
from askdocs.types import Chunk


class Reranker(Protocol):
    name: str

    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


class LexicalReranker:
    """A cheap reranker that needs no model.

    score = 0.65 * IDF-weighted query-term coverage
          + 0.25 * query-bigram coverage (rewards matching phrases, not just words)
          + 0.10 * coverage within the section header

    It's a baseline, not a cross-encoder. It catches 'v1 vs v2' style
    distractors because the distinguishing term carries high IDF.
    """

    name = "lite"

    def __init__(self, bm25: BM25Index):
        self.bm25 = bm25

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        q = list(dict.fromkeys(tokenize(query)))
        if not q:
            return [0.0] * len(chunks)
        weights = {t: self.bm25.term_idf(t) for t in q}
        total = sum(weights.values())
        q_bigrams = set(zip(q, q[1:], strict=False))
        out = []
        for c in chunks:
            body = tokenize(c.text)
            body_set = set(body)
            head_set = set(tokenize(c.header))
            cov = sum(w for t, w in weights.items() if t in body_set or t in head_set) / total
            bi = len(q_bigrams & set(zip(body, body[1:], strict=False))) / len(q_bigrams) if q_bigrams else cov
            head = sum(w for t, w in weights.items() if t in head_set) / total
            out.append(0.65 * cov + 0.25 * bi + 0.10 * head)
        return out


class CrossEncoderReranker:
    """ms-marco-MiniLM-L-6-v2 by default: about 22M params, runs on CPU in tens of ms for
    20 pairs. Logits go through a sigmoid so the gate threshold is meaningful."""

    def __init__(self, cfg: RerankConfig):
        from sentence_transformers import CrossEncoder  # optional dependency: pip install .[ml]

        self.name = cfg.model
        self._model = CrossEncoder(cfg.model)

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        logits = self._model.predict([(query, f"{c.header}\n{c.text}") for c in chunks], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def make_reranker(cfg: RerankConfig, bm25: BM25Index) -> Reranker | None:
    if cfg.provider == "none":
        return None
    if cfg.provider == "lite":
        return LexicalReranker(bm25)
    return CrossEncoderReranker(cfg)
