from askdocs.generation.citations import check_citations, support_score
from askdocs.obs.metrics import MetricsRegistry
from askdocs.obs.tracing import MemoryExporter, Tracer
from askdocs.pipeline import RAGPipeline
from askdocs.types import Chunk, Generation, Retrieved


def _ctx(*texts):
    return [Retrieved(Chunk(f"c{i}", "d.md", "Doc", "S", t), 1.0, "rerank") for i, t in enumerate(texts)]


def test_support_catches_wrong_number():
    src = ["The v1 API allows 500 publish requests per second."]
    assert support_score("The v1 API allows 500 publish requests per second [1].", src) == 1.0
    # Overlap alone is fooled by one wrong number in a copied sentence...
    assert support_score("The v1 API allows 900 publish requests per second [1].", src) >= 0.6
    # ...so numbers are a hard check.
    (s,) = check_citations("The v1 API allows 900 publish requests per second [1].", _ctx(*src), 0.6).sentences
    assert not s.supported and s.missing_numbers == ["900"]


def test_check_flags_missing_and_out_of_range_citations():
    ctx = _ctx("Retention defaults to 4 days.")
    rep = check_citations("Retention defaults to 4 days [1]. It is configurable [7]. Ask support.", ctx, 0.6)
    s1, s2, s3 = rep.sentences
    assert s1.supported
    assert not s2.supported and s2.invalid_citations == [7]
    assert not s3.supported and s3.citations == []
    assert not rep.ok and round(rep.coverage, 2) == 0.33


class ScriptedLLM:
    """Returns canned answers in order; records whether the retry carried feedback."""

    model = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.problems_seen = []

    def generate(self, question, contexts, problems=None):
        self.problems_seen.append(problems)
        return Generation(self.answers.pop(0), self.model, 100, 20)


def _pipe(settings, index, llm, **over):
    s = settings.model_copy(deep=True)
    for k, v in over.items():
        setattr(s.citations, k, v)
    exp = MemoryExporter()
    return RAGPipeline(s, index=index, tracer=Tracer([exp]), metrics=MetricsRegistry(), generator=llm), exp


Q = "What is the publish rate limit on the legacy API?"


def test_hallucination_triggers_retry_then_passes(settings, index):
    llm = ScriptedLLM(
        "The legacy API allows 9,999 publish requests per second [1].",  # wrong number
        "The legacy v1 API allows 500 publish requests per second [1].",
    )
    pipe, exp = _pipe(settings, index, llm)
    a = pipe.answer(Q)
    assert not a.abstained and "500" in a.text
    assert llm.problems_seen[0] is None and llm.problems_seen[1]  # retry carried the failure list
    assert a.quality["citation_retries"] == 1
    assert [s["name"] for s in exp.traces[0].to_dict()["spans"]].count("generate") == 2


def test_unsupported_sentence_is_dropped(settings, index):
    bad_twice = "The legacy v1 API allows 500 publish requests per second [1]. It also supports Kafka natively [1]."
    pipe, _ = _pipe(settings, index, ScriptedLLM(bad_twice, bad_twice))
    a = pipe.answer(Q)
    assert "500" in a.text and "Kafka" not in a.text
    assert a.quality["dropped_sentences"] == 1 and a.quality["support_ratio"] == 1.0


def test_fully_unsupported_answer_abstains(settings, index):
    junk = "Quantum entanglement powers the broker [1]."
    pipe, _ = _pipe(settings, index, ScriptedLLM(junk, junk))
    a = pipe.answer(Q)
    assert a.abstained and a.abstain_reason == "unsupported_answer"


def test_low_evidence_abstains_without_calling_llm(settings, index):
    llm = ScriptedLLM()  # would raise IndexError if called
    pipe, _ = _pipe(settings, index, llm)
    a = pipe.answer("Is LumenQueue SOC 2 Type II certified?")
    assert a.abstained and a.abstain_reason == "low_evidence" and a.usage["llm_calls"] == 0


def test_offline_pipeline_end_to_end_records_trace_and_metrics(settings, index):
    exp = MemoryExporter()
    reg = MetricsRegistry()
    pipe = RAGPipeline(settings, index=index, tracer=Tracer([exp]), metrics=reg)
    a = pipe.answer("What does error LQ-1033 mean?")
    assert not a.abstained and "allowlist" in a.text.lower() and a.citations
    t = exp.traces[0].to_dict()
    names = {s["name"] for s in t["spans"]}
    assert {"ask", "retrieve", "retrieve.bm25", "retrieve.dense", "fuse", "rerank", "gate", "generate", "citations.check"} <= names
    # every child points at a real parent
    ids = {s["span_id"] for s in t["spans"]}
    assert all(s["parent_id"] in ids for s in t["spans"][1:])
    assert a.timings_ms["ask"] >= a.timings_ms["retrieve"]
    prom = reg.render_prometheus()
    assert 'askdocs_requests_total{abstained="false"} 1' in prom
    assert "askdocs_stage_latency_ms_bucket" in prom and 'le="+Inf"' in prom
