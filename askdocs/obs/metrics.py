"""In-process metrics: counters and latency histograms, exposed in Prometheus
text format at GET /metrics. No client library is needed. The format is simple
and writing it by hand avoids a dependency."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

# Latency buckets in milliseconds: fine-grained at the low end where retrieval
# lives, coarse at the high end where LLM calls live.
DEFAULT_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


def percentile(values: list[float] | np.ndarray, p: float) -> float:
    """Linear-interpolated percentile (numpy's default). Returns 0.0 for empty input."""
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, p)) if arr.size else 0.0


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    esc = {k: str(v).replace("\\", "\\\\").replace('"', '\\"') for k, v in sorted(labels.items())}
    return "{" + ",".join(f'{k}="{v}"' for k, v in esc.items()) + "}"


class Histogram:
    def __init__(self, buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS, window: int = 10_000):
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.sum = 0.0
        self.n = 0
        self.recent: deque[float] = deque(maxlen=window)  # for exact p50/p95 on the dashboard

    def observe(self, v: float) -> None:
        self.n += 1
        self.sum += v
        self.recent.append(v)
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def quantile(self, p: float) -> float:
        return percentile(list(self.recent), p)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple], float] = {}
        self.hists: dict[tuple[str, tuple], Histogram] = {}
        self.help: dict[str, str] = {}

    def inc(self, name: str, value: float = 1.0, help: str = "", **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value
            if help:
                self.help.setdefault(name, help)

    def observe(self, name: str, value: float, help: str = "", **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.hists.setdefault(key, Histogram()).observe(value)
            if help:
                self.help.setdefault(name, help)

    def quantiles(self, name: str, ps: tuple[float, ...] = (50, 95, 99)) -> dict[str, dict[str, float]]:
        out = {}
        for (n, labels), h in self.hists.items():
            if n == name:
                out[",".join(f"{k}={v}" for k, v in labels) or "all"] = {f"p{int(p)}": round(h.quantile(p), 3) for p in ps}
        return out

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            names = sorted({n for n, _ in self.counters})
            for name in names:
                if name in self.help:
                    lines.append(f"# HELP {name} {self.help[name]}")
                lines.append(f"# TYPE {name} counter")
                for (n, labels), v in sorted(self.counters.items()):
                    if n == name:
                        lines.append(f"{name}{_labels(dict(labels))} {v:g}")
            for name in sorted({n for n, _ in self.hists}):
                if name in self.help:
                    lines.append(f"# HELP {name} {self.help[name]}")
                lines.append(f"# TYPE {name} histogram")
                for (n, labels), h in sorted(self.hists.items(), key=lambda kv: kv[0]):
                    if n != name:
                        continue
                    lab = dict(labels)
                    cum = 0
                    for b, c in zip(h.buckets, h.counts, strict=False):
                        cum += c
                        lines.append(f"{name}_bucket{_labels({**lab, 'le': f'{b:g}'})} {cum}")
                    lines.append(f"{name}_bucket{_labels({**lab, 'le': '+Inf'})} {h.n}")
                    lines.append(f"{name}_sum{_labels(lab)} {h.sum:g}")
                    lines.append(f"{name}_count{_labels(lab)} {h.n}")
        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()
