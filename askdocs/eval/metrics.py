"""Retrieval and answer metrics.

Relevance is defined by evidence strings, not chunk IDs: a chunk counts as
relevant to a question if it comes from the expected doc and contains the
evidence text. Chunk IDs change whenever chunking changes, so this is what
makes chunk-size ablations comparable.
"""

from __future__ import annotations

import math
import re

from askdocs.text import normalize
from askdocs.types import Chunk


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def chunk_matches(chunk: Chunk, evidence: dict) -> bool:
    return chunk.doc_id == evidence["doc"] and _ws(evidence["contains"]) in _ws(chunk.text)


def relevance_flags(ranked: list[Chunk], evidence: list[dict]) -> list[bool]:
    return [any(chunk_matches(c, e) for e in evidence) for c in ranked]


def recall_at_k(ranked: list[Chunk], evidence: list[dict], k: int) -> float:
    """Share of evidence items with at least one matching chunk in the top k."""
    if not evidence:
        return 0.0
    top = ranked[:k]
    return sum(1 for e in evidence if any(chunk_matches(c, e) for c in top)) / len(evidence)


def mrr_at_k(ranked: list[Chunk], evidence: list[dict], k: int) -> float:
    for i, rel in enumerate(relevance_flags(ranked[:k], evidence), start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[Chunk], evidence: list[dict], k: int, n_relevant_total: int) -> float:
    rels = relevance_flags(ranked[:k], evidence)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, max(n_relevant_total, 1))))
    return dcg / ideal if ideal else 0.0


def _norm_answer(s: str) -> str:
    return re.sub(r"\s+", " ", normalize(s.replace("`", "")))


def keyword_recall(answer: str, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    a = _norm_answer(answer)
    return sum(1 for k in must_include if _norm_answer(k) in a) / len(must_include)


def citation_precision(cited: list[Chunk], evidence: list[dict]) -> float | None:
    """Share of cited chunks that contain gold evidence. None if nothing was cited."""
    if not cited:
        return None
    return sum(1 for c in cited if any(chunk_matches(c, e) for e in evidence)) / len(cited)
