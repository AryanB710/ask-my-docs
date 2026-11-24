"""Small tracing layer: nested spans with timings and attributes.

Each finished trace is written as one JSON line (runs/traces/*.jsonl), which is
what the dashboard and `askdocs report` read. If observability.otel is on and
the OpenTelemetry SDK is installed, the same spans also go to an OTLP collector
(Jaeger, Tempo, Honeycomb and so on) without touching pipeline code.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


@dataclass
class Trace:
    trace_id: str
    name: str
    started_at: float  # wall clock, for the report
    spans: list[Span] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def stage_ms(self) -> dict[str, float]:
        """Duration of each named span. Repeated names (e.g. a regenerate step) are summed."""
        out: dict[str, float] = {}
        for s in self.spans:
            out[s.name] = out.get(s.name, 0.0) + s.duration_ms
        return {k: round(v, 3) for k, v in out.items()}

    def to_dict(self) -> dict[str, Any]:
        root = self.spans[0] if self.spans else None
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "ts": self.started_at,
            "duration_ms": round(root.duration_ms, 3) if root else 0.0,
            "status": root.status if root else "ok",
            "attributes": self.attributes,
            "stages_ms": self.stage_ms(),
            "spans": [
                {
                    "name": s.name,
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "duration_ms": round(s.duration_ms, 3),
                    "status": s.status,
                    "error": s.error,
                    "attributes": s.attributes,
                }
                for s in self.spans
            ],
        }


class Exporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class JsonlExporter:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.dir / f"traces-{time.strftime('%Y%m%d', time.gmtime())}.jsonl"

    def export(self, trace: Trace) -> None:
        line = json.dumps(trace.to_dict(), default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class MemoryExporter:
    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def export(self, trace: Trace) -> None:
        self.traces.append(trace)


class OTelBridge:
    """Replays finished spans into OpenTelemetry. Needs opentelemetry-sdk and
    opentelemetry-exporter-otlp (pip install .[otel]). Reads OTEL_EXPORTER_OTLP_ENDPOINT."""

    def __init__(self, service_name: str):
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        otel_trace.set_tracer_provider(provider)
        self._otel = otel_trace
        self._tracer = otel_trace.get_tracer("askdocs")

    def export(self, trace: Trace) -> None:
        # Rebuild the tree with real start/end timestamps (ns since epoch).
        base_ns = int(trace.started_at * 1e9)
        t0 = trace.spans[0].start if trace.spans else 0.0
        live: dict[str, Any] = {}
        for s in trace.spans:
            parent = live.get(s.parent_id) if s.parent_id else None
            ctx = self._otel.set_span_in_context(parent) if parent is not None else None
            start_ns = base_ns + int((s.start - t0) * 1e9)
            otel_span = self._tracer.start_span(s.name, context=ctx, start_time=start_ns)
            for k, v in s.attributes.items():
                if isinstance(v, (str, bool, int, float)):
                    otel_span.set_attribute(k, v)
            live[s.span_id] = otel_span
        for s in reversed(trace.spans):
            end_ns = base_ns + int(((s.end or s.start) - t0) * 1e9)
            live[s.span_id].end(end_time=end_ns)


_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("askdocs_trace", default=None)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("askdocs_span", default=None)


class Tracer:
    def __init__(self, exporters: list[Exporter] | None = None):
        self.exporters = exporters or []

    @contextmanager
    def trace(self, name: str, **attrs: Any) -> Iterator[Trace]:
        tr = Trace(trace_id=uuid.uuid4().hex, name=name, started_at=time.time(), attributes=dict(attrs))
        tok_t = _current_trace.set(tr)
        try:
            with self.span(name):
                yield tr
        finally:
            _current_trace.reset(tok_t)
            for e in self.exporters:
                try:
                    e.export(tr)
                except Exception:  # never let telemetry break a request
                    pass

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        tr = _current_trace.get()
        parent = _current_span.get()
        sp = Span(
            name=name,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start=time.perf_counter(),
            attributes=dict(attrs),
        )
        if tr is not None:
            tr.spans.append(sp)
        tok = _current_span.set(sp)
        try:
            yield sp
        except Exception as exc:
            sp.status, sp.error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.end = time.perf_counter()
            _current_span.reset(tok)


def make_tracer(
    trace_dir: str | None, otel: bool = False, service_name: str = "ask-my-docs", extra: list[Exporter] | None = None
) -> Tracer:
    exporters: list[Exporter] = list(extra or [])
    if trace_dir:
        exporters.append(JsonlExporter(trace_dir))
    if otel:
        try:
            exporters.append(OTelBridge(service_name))
        except ImportError:
            import logging

            logging.getLogger(__name__).warning("observability.otel=true but opentelemetry-sdk is not installed; skipping OTLP export")
    return Tracer(exporters)
