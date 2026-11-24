"""The request path, in one place:

question
  -> retrieve (BM25 + dense -> RRF -> rerank -> top-k)
  -> evidence gate       (abstain early if the best chunk is weak; no LLM call)
  -> generate            (LLM with numbered sources)
  -> citation check      (regenerate once if unsupported, then drop or abstain)
  -> answer + trace + metrics
"""

from __future__ import annotations

import logging

from askdocs.config import Settings
from askdocs.generation.citations import (
    check_citations,
    citation_payload,
    keep_supported,
    problems_for_retry,
)
from askdocs.generation.llm import Generator, make_generator
from askdocs.generation.prompts import ABSTAIN_TOKEN
from askdocs.ingest.index import Index, load_or_build
from askdocs.obs.cost import request_cost
from askdocs.obs.metrics import REGISTRY, MetricsRegistry
from askdocs.obs.tracing import Tracer, make_tracer
from askdocs.retrieval.retriever import HybridRetriever
from askdocs.types import Answer, CitationReport

log = logging.getLogger(__name__)

ABSTAIN_MESSAGE = "I couldn't find this in the documentation."


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        index: Index | None = None,
        tracer: Tracer | None = None,
        metrics: MetricsRegistry | None = None,
        generator: Generator | None = None,
    ):
        self.settings = settings
        self.index = index or load_or_build(settings)
        obs = settings.observability
        self.tracer = tracer or make_tracer(obs.trace_dir, obs.otel, obs.service_name)
        self.metrics = metrics or REGISTRY
        self.retriever = HybridRetriever(self.index, settings, self.tracer)
        self.generator = generator or make_generator(settings.llm, self.index.bm25)

    def _abstain(self, question, reason, contexts, tr, usage, quality) -> Answer:
        self.metrics.inc("askdocs_abstentions_total", help="Answers withheld", reason=reason)
        tr.attributes.update(abstained=True, abstain_reason=reason)
        return Answer(question, ABSTAIN_MESSAGE, True, reason, [], contexts, tr.trace_id, tr.stage_ms(), usage, quality)

    def answer(self, question: str) -> Answer:
        s = self.settings
        usage = {"model": self.generator.model, "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0, "cost_usd": 0.0}
        quality: dict = {}
        with self.tracer.trace("ask", question=question, config=s.name) as tr:
            try:
                result = self._run(question, tr, usage, quality)
            except Exception:
                self.metrics.inc("askdocs_errors_total", help="Requests that raised")
                raise
            finally:
                usage["cost_usd"] = round(
                    request_cost(usage["model"], usage["prompt_tokens"], usage["completion_tokens"], s.observability.pricing_file), 8
                )
                tr.attributes.update(usage=usage, quality=quality)
        # Re-read timings after the root span closed so "ask" is the full request.
        result.timings_ms = tr.stage_ms()
        self._record(result)
        return result

    def _run(self, question: str, tr, usage: dict, quality: dict) -> Answer:
        s = self.settings
        with self.tracer.span("retrieve") as sp:
            rr = self.retriever.retrieve(question)
            sp.set(k=len(rr.contexts), evidence_score=round(rr.evidence_score, 4))
        contexts = rr.contexts
        quality["evidence_score"] = round(rr.evidence_score, 4)
        quality["retrieved"] = [r.chunk.chunk_id for r in contexts]
        quality["ranked"] = [r.chunk.chunk_id for r in rr.ranked[:20]]

        with self.tracer.span("gate") as sp:
            passed = bool(contexts) and rr.evidence_score >= s.gate.min_evidence_score
            sp.set(threshold=s.gate.min_evidence_score, passed=passed)
        if not passed:
            return self._abstain(question, "low_evidence", contexts, tr, usage, quality)

        problems: list[str] | None = None
        report: CitationReport | None = None
        text = ""
        for attempt in range(s.citations.max_regenerations + 1):
            with self.tracer.span("generate", attempt=attempt) as sp:
                gen = self.generator.generate(question, contexts, problems)
                usage["llm_calls"] += 1
                usage["prompt_tokens"] += gen.prompt_tokens
                usage["completion_tokens"] += gen.completion_tokens
                sp.set(model=gen.model, prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens)
            text = gen.text.strip()
            if text == ABSTAIN_TOKEN or ABSTAIN_TOKEN in text:
                quality["citation_retries"] = attempt
                return self._abstain(question, "model_declined", contexts, tr, usage, quality)
            if not s.citations.enforce:
                report = check_citations(text, contexts, s.citations.min_support)  # measured, not enforced
                break
            with self.tracer.span("citations.check", attempt=attempt) as sp:
                report = check_citations(text, contexts, s.citations.min_support)
                sp.set(coverage=round(report.coverage, 3), support_ratio=round(report.support_ratio, 3), ok=report.ok)
            if report.ok:
                break
            problems = problems_for_retry(report)
            self.metrics.inc("askdocs_citation_failures_total", help="Generations that failed the citation check")

        assert report is not None
        quality["citation_retries"] = attempt
        quality["citation_coverage_raw"] = round(report.coverage, 3)
        quality["support_ratio_raw"] = round(report.support_ratio, 3)
        quality["sentences"] = [
            {"text": c.sentence, "citations": c.citations, "support": c.support, "supported": c.supported} for c in report.sentences
        ]

        if s.citations.enforce and not report.ok:
            if s.citations.on_failure == "abstain" or not keep_supported(report):
                return self._abstain(question, "unsupported_answer", contexts, tr, usage, quality)
            dropped = sum(1 for c in report.sentences if not c.supported)
            quality["dropped_sentences"] = dropped
            text = keep_supported(report)
            report = check_citations(text, contexts, s.citations.min_support)

        quality["citation_coverage"] = round(report.coverage, 3)
        quality["support_ratio"] = round(report.support_ratio, 3)
        citations = citation_payload(report, contexts, only_supported=s.citations.enforce)
        tr.attributes.update(abstained=False)
        return Answer(question, text, False, None, citations, contexts, tr.trace_id, tr.stage_ms(), usage, quality)

    def _record(self, a: Answer) -> None:
        m = self.metrics
        m.inc("askdocs_requests_total", help="Questions answered or abstained", abstained=str(a.abstained).lower())
        m.observe("askdocs_request_latency_ms", a.timings_ms.get("ask", 0.0), help="End-to-end latency (ms)")
        for stage, ms in a.timings_ms.items():
            if stage != "ask":
                m.observe("askdocs_stage_latency_ms", ms, help="Per-stage latency (ms)", stage=stage)
        m.inc("askdocs_tokens_total", a.usage["prompt_tokens"], help="LLM tokens", kind="prompt")
        m.inc("askdocs_tokens_total", a.usage["completion_tokens"], kind="completion")
        m.inc("askdocs_cost_usd_total", a.usage["cost_usd"], help="Estimated LLM spend (USD)")
        if "support_ratio" in a.quality:
            m.observe(
                "askdocs_support_ratio_pct", 100 * a.quality["support_ratio"], help="Share of answer sentences supported by citations (%)"
            )
        m.observe("askdocs_evidence_score_pct", 100 * a.quality.get("evidence_score", 0.0), help="Top reranker score (%)")
