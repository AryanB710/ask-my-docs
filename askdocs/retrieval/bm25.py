"""Okapi BM25 over an inverted index, written from scratch (about 60 lines) instead of
pulling in rank_bm25. The scoring is easy to reason about, and the IDF table is
reused by the lite reranker and the extractive generator."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from askdocs.text import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(texts)
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        # BM25+ style non-negative IDF so very common terms never subtract score.
        self.idf = {t: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self._max_idf = max(self.idf.values()) if self.idf else 1.0

    def term_idf(self, term: str) -> float:
        # Unseen terms get the max IDF: a query word that appears nowhere is maximally
        # informative, and that's what makes out-of-scope questions score low.
        return self.idf.get(term, self._max_idf)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idx = np.fromiter((i for i, _ in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((f for _, f in plist), dtype=np.float32, count=len(plist))
            out[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return out

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        if not np.any(s > 0):
            return []
        k = min(k, int((s > 0).sum()))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(int(i), float(s[i])) for i in top]
