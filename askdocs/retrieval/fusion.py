"""Reciprocal Rank Fusion (Cormack et al., 2009).

Why RRF instead of a weighted sum of scores: BM25 scores are unbounded and
cosine similarities sit in [-1, 1], so mixing them needs per-corpus
normalisation that breaks whenever the corpus changes. RRF only uses ranks, so
it has one parameter (k=60) that rarely needs tuning.
"""

from __future__ import annotations


def rrf(rankings: dict[str, list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float, dict[str, int]]]:
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ranked in rankings.items():
        for rank, (idx, _score) in enumerate(ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(idx, {})[name] = rank
    order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(idx, score, ranks[idx]) for idx, score in order]
