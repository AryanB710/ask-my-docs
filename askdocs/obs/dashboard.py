"""Reads trace JSONL files and summarises them: latency percentiles (total and
per stage), cost, abstentions, citation quality, errors. Renders a
self-contained HTML dashboard with no JS dependencies, served at GET /dashboard
and written by `askdocs report --html`."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from askdocs.obs.metrics import percentile


def load_traces(path: str | Path, limit: int | None = 5000) -> list[dict]:
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p] if p.exists() else []
    traces: list[dict] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    traces.sort(key=lambda t: t.get("ts", 0))
    return traces[-limit:] if limit else traces


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def summarize(traces: list[dict]) -> dict:
    asks = [t for t in traces if t.get("name") == "ask"]
    total = [t["duration_ms"] for t in asks]
    stages: dict[str, list[float]] = {}
    for t in asks:
        for k, v in t.get("stages_ms", {}).items():
            if k not in ("ask", "retrieve"):  # retrieve is the parent of the retrieve.* spans
                stages.setdefault(k, []).append(v)
    usage = [t.get("attributes", {}).get("usage", {}) for t in asks]
    quality = [t.get("attributes", {}).get("quality", {}) for t in asks]
    abstained = [t.get("attributes", {}).get("abstained", False) for t in asks]
    reasons = Counter(t["attributes"].get("abstain_reason") for t in asks if t.get("attributes", {}).get("abstained"))
    costs = [u.get("cost_usd", 0.0) for u in usage]
    return {
        "requests": len(asks),
        "errors": sum(1 for t in asks if t.get("status") == "error"),
        "latency_ms": {
            "p50": percentile(total, 50),
            "p95": percentile(total, 95),
            "p99": percentile(total, 99),
            "max": max(total, default=0.0),
        },
        "stages_ms": {
            k: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)}
            for k, v in sorted(stages.items(), key=lambda kv: -percentile(kv[1], 95))
        },
        "cost_usd": {"total": sum(costs), "per_request": _mean(costs), "per_1k": 1000 * _mean(costs)},
        "tokens": {
            "prompt_avg": _mean([u.get("prompt_tokens") for u in usage]),
            "completion_avg": _mean([u.get("completion_tokens") for u in usage]),
        },
        "abstain_rate": _mean([1.0 if a else 0.0 for a in abstained]),
        "abstain_reasons": dict(reasons),
        "support_ratio_avg": _mean([q.get("support_ratio") for q in quality]),
        "evidence_score_avg": _mean([q.get("evidence_score") for q in quality]),
        "citation_retry_rate": _mean([1.0 if q.get("citation_retries") else 0.0 for q in quality]),
        "latency_samples": total,
        "recent": [
            {
                "ts": t.get("ts"),
                "question": t.get("attributes", {}).get("question", ""),
                "ms": t["duration_ms"],
                "abstained": t.get("attributes", {}).get("abstained", False),
                "support": t.get("attributes", {}).get("quality", {}).get("support_ratio"),
                "cost": t.get("attributes", {}).get("usage", {}).get("cost_usd", 0.0),
                "status": t.get("status"),
                "trace_id": t.get("trace_id"),
            }
            for t in asks[-25:][::-1]
        ],
    }


_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;--text-secondary:#52514e;
--text-muted:#7a7974;--grid:#e4e3df;--series-1:#2a78d6;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;
background:var(--surface-1);color:var(--text-primary);margin:0;padding:28px;max-width:1080px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--surface-2:#242422;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e86;--grid:#33332f;--series-1:#3987e5}}
body{margin:0;background:var(--surface-1)}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 10px}
.sub{color:var(--text-secondary);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface-2);border-radius:10px;padding:14px 16px}
.tile .v{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .k{color:var(--text-secondary);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .n{color:var(--text-muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:500;font-size:12px}
td.num,th.num{text-align:right}
svg text{fill:var(--text-secondary);font-size:12px}
svg .bar{fill:var(--series-1)} svg .bar:hover{opacity:.8}
svg .grid{stroke:var(--grid);stroke-width:1}
details{margin-top:8px;color:var(--text-secondary)}
"""


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Horizontal bar anchored at x, rounded only on the data end."""
    r = min(r, w / 2, h / 2)
    if w <= 0:
        return ""
    return f"M{x},{y}h{w - r}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r}a{r},{r} 0 0 1 -{r},{r}h-{w - r}z"


def _stage_chart(stages: dict) -> str:
    if not stages:
        return "<p class='sub'>No stage data yet.</p>"
    rows = list(stages.items())
    label_w, chart_w, row_h, bar_h = 150, 560, 30, 14
    vmax = max(v["p95"] for _, v in rows) or 1.0
    h = row_h * len(rows) + 24
    parts = [f'<svg width="{label_w + chart_w + 120}" height="{h}" role="img" aria-label="p95 latency by pipeline stage">']
    for i in range(5):
        gx = label_w + chart_w * i / 4
        parts.append(f'<line class="grid" x1="{gx}" x2="{gx}" y1="0" y2="{h - 20}"/>')
        parts.append(f'<text x="{gx}" y="{h - 4}" text-anchor="middle">{vmax * i / 4:.1f}{" ms" if i == 4 else ""}</text>')
    for i, (name, v) in enumerate(rows):
        y = i * row_h + (row_h - bar_h) / 2
        w = chart_w * v["p95"] / vmax
        parts.append(f'<text x="{label_w - 10}" y="{y + bar_h - 2}" text-anchor="end">{html.escape(name)}</text>')
        parts.append(
            f'<path class="bar" d="{_bar_path(label_w, y, max(w, 1), bar_h)}">'
            f"<title>{html.escape(name)}: p50 {v['p50']:.2f} ms · p95 {v['p95']:.2f} ms · {v['n']} spans</title></path>"
        )
        parts.append(f'<text x="{label_w + w + 8}" y="{y + bar_h - 2}">{v["p95"]:.2f} ms</text>')
    parts.append("</svg>")
    return "".join(parts)


def _histogram(samples: list[float], bins: int = 24) -> str:
    if len(samples) < 2:
        return "<p class='sub'>Need at least two requests for a distribution.</p>"
    lo, hi = 0.0, percentile(samples, 99) or max(samples)
    width = (hi - lo) / bins or 1.0
    counts = [0] * bins
    for s in samples:
        counts[min(bins - 1, int((min(s, hi) - lo) / width))] += 1
    cw, ch, gap = 760, 160, 2
    bw = cw / bins
    cmax = max(counts) or 1
    p50, p95 = percentile(samples, 50), percentile(samples, 95)
    parts = [f'<svg width="{cw + 40}" height="{ch + 40}" role="img" aria-label="End-to-end latency distribution">']
    parts.append(f'<line class="grid" x1="0" x2="{cw}" y1="{ch}" y2="{ch}"/>')
    for i, c in enumerate(counts):
        if not c:
            continue
        bh = ch * c / cmax
        x = i * bw + gap / 2
        w = bw - gap
        r = min(4, w / 2, bh / 2)
        d = f"M{x},{ch}v-{bh - r}a{r},{r} 0 0 1 {r},-{r}h{w - 2 * r}a{r},{r} 0 0 1 {r},{r}v{bh - r}z"
        parts.append(f'<path class="bar" d="{d}"><title>{lo + i * width:.1f}–{lo + (i + 1) * width:.1f} ms: {c} requests</title></path>')
    # p50 label sits left of its line and p95 to the right, so they never collide
    for label, v, anchor, dx in (("p50", p50, "end", -4), ("p95", p95, "start", 4)):
        x = cw * (v - lo) / (hi - lo or 1)
        parts.append(f'<line x1="{x}" x2="{x}" y1="0" y2="{ch}" stroke="currentColor" stroke-dasharray="3 3" opacity=".6"/>')
        parts.append(f'<text x="{x + dx}" y="12" text-anchor="{anchor}">{label} {v:.1f} ms</text>')
    parts.append(f'<text x="0" y="{ch + 16}">0 ms</text><text x="{cw}" y="{ch + 16}" text-anchor="end">{hi:.1f} ms (p99)</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_dashboard(summary: dict) -> str:
    s = summary
    lat, cost = s["latency_ms"], s["cost_usd"]
    tiles = [
        ("Requests", f"{s['requests']}", f"{s['errors']} errors"),
        ("p50 latency", f"{lat['p50']:.1f} ms", "end to end"),
        ("p95 latency", f"{lat['p95']:.1f} ms", f"p99 {lat['p99']:.1f} ms"),
        ("Cost / 1k req", f"${cost['per_1k']:.4f}", f"${cost['total']:.4f} total"),
        ("Abstain rate", f"{s['abstain_rate']:.1%}", ", ".join(f"{k} {v}" for k, v in s["abstain_reasons"].items()) or "—"),
        ("Supported sentences", f"{s['support_ratio_avg']:.1%}", f"retry rate {s['citation_retry_rate']:.1%}"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{html.escape(n)}</div></div>'
        for k, v, n in tiles
    )
    stage_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v['p50']:.2f}</td><td class='num'>{v['p95']:.2f}</td><td class='num'>{v['n']}</td></tr>"
        for k, v in s["stages_ms"].items()
    )

    def _row(r: dict) -> str:
        support = "" if r["support"] is None else f"{r['support']:.0%}"
        outcome = ("abstained" if r["abstained"] else "answered") + (" · error" if r["status"] == "error" else "")
        return (
            f"<tr><td>{html.escape(r['question'][:90])}</td><td class='num'>{r['ms']:.1f}</td><td>{outcome}</td>"
            f"<td class='num'>{support}</td><td class='num'>${r['cost']:.5f}</td><td><code>{(r['trace_id'] or '')[:8]}</code></td></tr>"
        )

    recent = "".join(_row(r) for r in s["recent"])
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ask-my-docs · observability</title><style>{_CSS}</style></head><body><main class="viz-root">
<h1>ask-my-docs · observability</h1><p class="sub">Last {s["requests"]} requests from trace logs. Refresh to update.</p>
<div class="tiles">{tile_html}</div>
<h2>p95 latency by stage</h2>{_stage_chart(s["stages_ms"])}
<details><summary>Table view</summary><table><thead><tr><th>Stage</th><th class="num">p50 ms</th><th class="num">p95 ms</th><th class="num">spans</th></tr></thead><tbody>{stage_rows}</tbody></table></details>
<h2>End-to-end latency distribution</h2>{_histogram(s["latency_samples"])}
<h2>Recent requests</h2><table><thead><tr><th>Question</th><th class="num">ms</th><th>Outcome</th><th class="num">Supported</th><th class="num">Cost</th><th>Trace</th></tr></thead><tbody>{recent}</tbody></table>
</main></body></html>"""


def write_dashboard(summary: dict, traces: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(summary), encoding="utf-8")
