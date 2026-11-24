import json

import pytest

from askdocs.obs.dashboard import load_traces, render_dashboard, summarize
from askdocs.obs.tracing import JsonlExporter, MemoryExporter, Tracer
from askdocs.pipeline import RAGPipeline


def test_span_error_is_recorded_and_reraised():
    exp = MemoryExporter()
    tr = Tracer([exp])
    with pytest.raises(ValueError):
        with tr.trace("ask"):
            with tr.span("generate"):
                raise ValueError("boom")
    spans = exp.traces[0].to_dict()["spans"]
    assert spans[1]["status"] == "error" and "boom" in spans[1]["error"]
    assert spans[0]["status"] == "error"


def test_jsonl_traces_feed_dashboard(tmp_path, settings, index):
    pipe = RAGPipeline(settings, index=index, tracer=Tracer([JsonlExporter(tmp_path)]))
    for q in ["What does error LQ-1033 mean?", "Is LumenQueue SOC 2 Type II certified?", "What is the default visibility timeout?"]:
        pipe.answer(q)
    traces = load_traces(tmp_path)
    assert len(traces) == 3 and json.dumps(traces)  # serialisable
    s = summarize(traces)
    assert s["requests"] == 3 and 0 < s["abstain_rate"] < 1
    assert s["latency_ms"]["p95"] >= s["latency_ms"]["p50"] > 0
    assert "rerank" in s["stages_ms"]
    html = render_dashboard(s)
    assert "<svg" in html and "p95 latency by stage" in html
