"""HTTP API.

  POST /ask        {"question": "..."}  -> answer, citations, timings, cost, trace_id
  POST /feedback   {"trace_id": "...", "helpful": true, "comment": "..."}
  GET  /metrics    Prometheus text format (scrape it, or open it in a browser)
  GET  /dashboard  HTML latency/cost/quality dashboard built from the trace logs
  GET  /healthz    liveness + index info
  GET  /           minimal chat UI (for the demo GIF)

Run:  askdocs serve --config configs/local.yaml
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from askdocs.config import load_settings
from askdocs.obs.dashboard import load_traces, render_dashboard, summarize
from askdocs.obs.metrics import REGISTRY
from askdocs.pipeline import RAGPipeline

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    include_contexts: bool = False


class FeedbackRequest(BaseModel):
    trace_id: str
    helpful: bool
    comment: str | None = Field(default=None, max_length=2000)


_state: dict = {}
_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    with _lock:
        if "pipe" not in _state:
            overrides = json.loads(os.environ.get("ASKDOCS_SET", "[]"))
            _state["pipe"] = RAGPipeline(load_settings(os.environ.get("ASKDOCS_CONFIG", "configs/offline.yaml"), overrides))
        return _state["pipe"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_pipeline()  # build/load the index at startup, not on the first user request
    yield


app = FastAPI(title="ask-my-docs", version="0.1.0", lifespan=lifespan)


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    pipe = get_pipeline()
    try:
        # The pipeline is sync and CPU/IO bound, so keep it off the event loop.
        answer = await run_in_threadpool(pipe.answer, req.question.strip())
    except Exception as e:  # surface as 502: the upstream (LLM/embedder) is usually the cause
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
    return answer.to_dict(include_contexts=req.include_contexts)


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    pipe = get_pipeline()
    REGISTRY.inc("askdocs_feedback_total", help="User feedback on answers", helpful=str(req.helpful).lower())
    trace_dir = pipe.settings.observability.trace_dir
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(trace_dir) / "feedback.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **req.model_dump()}) + "\n")
    return {"ok": True}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return REGISTRY.render_prometheus()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    trace_dir = get_pipeline().settings.observability.trace_dir
    traces = load_traces(trace_dir) if trace_dir else []
    return render_dashboard(summarize(traces))


@app.get("/healthz")
async def healthz() -> dict:
    pipe = get_pipeline()
    return {"status": "ok", "config": pipe.settings.name, "index": pipe.index.manifest, "generator": pipe.generator.model}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
