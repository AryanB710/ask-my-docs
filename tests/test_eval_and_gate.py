import copy

from askdocs.eval.gate import evaluate_gate
from askdocs.eval.metrics import keyword_recall, mrr_at_k, ndcg_at_k, recall_at_k
from askdocs.obs.metrics import percentile
from askdocs.types import Chunk

EV = [{"doc": "a.md", "contains": "the answer is 42"}]


def _c(doc, text):
    return Chunk(text[:5], doc, "T", "S", text)


def test_retrieval_metrics():
    ranked = [_c("b.md", "noise"), _c("a.md", "so the answer is 42 here"), _c("a.md", "other")]
    assert recall_at_k(ranked, EV, 1) == 0.0 and recall_at_k(ranked, EV, 2) == 1.0
    assert mrr_at_k(ranked, EV, 10) == 0.5
    assert 0.6 < ndcg_at_k(ranked, EV, 10, n_relevant_total=1) < 0.64  # 1/log2(3)


def test_relevance_requires_matching_doc():
    assert recall_at_k([_c("other.md", "the answer is 42")], EV, 5) == 0.0


def test_keyword_recall_normalizes_commas_and_case():
    assert keyword_recall("It allows 3000 requests", ["3,000"]) == 1.0
    assert keyword_recall("Uses .FIFO suffix", [".fifo", "missing"]) == 0.5


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile(list(range(1, 101)), 50) == 50.5


def _results(**m):
    base = {"recall@5": 0.95, "answer_accuracy": 0.8, "latency_p95_ms": 10.0, "cost_per_1k_usd": 0.0, "false_abstain_rate": 0.05}
    base.update(m)
    return {"primary": "p", "variants": {"p": {"metrics": base}}}


TH = {
    "absolute": {"recall@5": {"min": 0.9}, "latency_p95_ms": {"max": 100}},
    "regression": {"max_quality_drop": 0.02, "max_latency_increase_pct": 50, "latency_floor_ms": 0},
}


def test_gate_passes_on_identical_run():
    r = _results()
    assert all(c.passed for c in evaluate_gate(r, TH, copy.deepcopy(r)))


def test_gate_fails_on_quality_regression_even_above_floor():
    checks = evaluate_gate(_results(answer_accuracy=0.75), TH, _results())
    failed = {c.metric for c in checks if not c.passed}
    assert failed == {"answer_accuracy"}


def test_gate_fails_on_floor_and_latency():
    checks = evaluate_gate(_results(**{"recall@5": 0.85, "latency_p95_ms": 16.0}), TH, _results())
    failed = {(c.metric, c.kind) for c in checks if not c.passed}
    assert ("recall@5", "floor") in failed and ("latency_p95_ms", "regression") in failed


def test_gate_flags_false_abstain_increase():
    checks = evaluate_gate(_results(false_abstain_rate=0.10), TH, _results())
    assert any(c.metric == "false_abstain_rate" and not c.passed for c in checks)
