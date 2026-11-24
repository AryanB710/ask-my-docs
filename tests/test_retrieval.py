from askdocs.obs.tracing import Tracer
from askdocs.retrieval.bm25 import BM25Index
from askdocs.retrieval.fusion import rrf
from askdocs.retrieval.retriever import HybridRetriever


def test_bm25_prefers_exact_identifier():
    idx = BM25Index(["error LQ-1003 means role denied", "error LQ-1033 means ip not in allowlist", "billing"])
    top = idx.search("what is LQ-1033", 3)
    assert top[0][0] == 1
    assert idx.search("kafka protocol", 3) == []


def test_rrf_rewards_agreement():
    fused = rrf({"a": [(1, 9.0), (2, 8.0), (3, 7.0)], "b": [(2, 0.9), (4, 0.8), (1, 0.7)]}, k=60)
    order = [i for i, _, _ in fused]
    assert order[:2] == [2, 1]  # in both lists, near the top
    assert fused[0][2] == {"a": 2, "b": 1}


def test_hybrid_retriever_finds_v1_not_v2(settings, index):
    r = HybridRetriever(index, settings, Tracer([]))
    res = r.retrieve("What is the publish rate limit on the legacy API?")
    assert "v1" in res.contexts[0].chunk.section
    assert 0.0 <= res.evidence_score <= 1.0
    assert set(res.contexts[0].ranks) >= {"bm25", "dense"} or "fused" in res.contexts[0].ranks


def test_out_of_scope_question_scores_low(settings, index):
    r = HybridRetriever(index, settings, Tracer([]))
    assert r.retrieve("Does LumenQueue support the Kafka wire protocol?").evidence_score < settings.gate.min_evidence_score
