"""Retrieval orchestration: BM25 and/or dense, then RRF, then rerank, then top-k.

Every stage runs inside its own span, so the trace shows exactly where the time
goes and which retriever found each chunk (r.ranks = {"bm25": 3, "dense": 1})."""

from __future__ import annotations

from dataclasses import dataclass

from askdocs.config import Settings
from askdocs.ingest.index import Index
from askdocs.obs.tracing import Tracer
from askdocs.retrieval.fusion import rrf
from askdocs.retrieval.rerank import LexicalReranker, make_reranker
from askdocs.types import Retrieved


@dataclass
class RetrievalResult:
    contexts: list[Retrieved]
    ranked: list[Retrieved]  # full final ordering (reranked pool + the rest), used for MRR/nDCG
    evidence_score: float


class HybridRetriever:
    def __init__(self, index: Index, settings: Settings, tracer: Tracer):
        self.index = index
        self.cfg = settings.retrieval
        self.rcfg = settings.reranker
        self.tracer = tracer
        self.reranker = make_reranker(self.rcfg, index.bm25)
        # Used only to compute an evidence score for the abstention gate when no reranker is configured.
        self._fallback_scorer = LexicalReranker(index.bm25)
        if self.cfg.mode in ("dense", "hybrid") and index.dense is None:
            raise ValueError(f"retrieval.mode={self.cfg.mode} needs a dense index")

    def retrieve(self, query: str) -> RetrievalResult:
        rankings: dict[str, list[tuple[int, float]]] = {}
        if self.cfg.mode in ("bm25", "hybrid"):
            with self.tracer.span("retrieve.bm25") as sp:
                rankings["bm25"] = self.index.bm25.search(query, self.cfg.bm25_k)
                sp.set(hits=len(rankings["bm25"]))
        if self.cfg.mode in ("dense", "hybrid"):
            with self.tracer.span("embed.query") as sp:
                qv = self.index.dense.embedder.embed_query(query)
                sp.set(model=self.index.dense.embedder.name)
            with self.tracer.span("retrieve.dense") as sp:
                rankings["dense"] = self.index.dense.search_vector(qv, self.cfg.dense_k)
                sp.set(hits=len(rankings["dense"]))

        with self.tracer.span("fuse") as sp:
            if len(rankings) == 1:
                ((name, single),) = rankings.items()
                fused = [(i, s, {name: r}) for r, (i, s) in enumerate(single, start=1)]
            else:
                fused = rrf(rankings, k=self.cfg.rrf_k)
            sp.set(method="rrf" if len(rankings) > 1 else "single", candidates=len(fused))
            candidates = [
                Retrieved(self.index.chunks[i], score, "hybrid" if len(rankings) > 1 else next(iter(rankings)), ranks)
                for i, score, ranks in fused
            ]

        if self.reranker is not None:
            pool = candidates[: self.rcfg.top_n]
            with self.tracer.span("rerank") as sp:
                scores = self.reranker.score(query, [r.chunk for r in pool])
                for fused_rank, (r, s) in enumerate(zip(pool, scores, strict=True), start=1):
                    r.ranks["fused"] = fused_rank
                    r.score, r.source = float(s), "rerank"
                pool.sort(key=lambda r: -r.score)
                sp.set(model=self.reranker.name, pool=len(pool))
            ranked = pool + candidates[self.rcfg.top_n :]
            evidence = pool[0].score if pool else 0.0
        else:
            ranked = candidates
            evidence = max(self._fallback_scorer.score(query, [r.chunk for r in ranked[: self.cfg.final_k]]), default=0.0)
        return RetrievalResult(contexts=ranked[: self.cfg.final_k], ranked=ranked, evidence_score=float(evidence))
