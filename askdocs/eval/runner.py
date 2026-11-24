"""Runs the golden set against one or more pipeline variants (an ablation suite)
and writes:

  eval/results/latest.json         summary metrics per variant (what the gate reads)
  eval/results/latest.md           the same as a markdown table (PR comment, README)
  eval/results/per_question.jsonl  every answer, for debugging regressions
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from askdocs.config import load_settings
from askdocs.eval.metrics import (
    chunk_matches,
    citation_precision,
    keyword_recall,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from askdocs.ingest.index import load_or_build
from askdocs.obs.metrics import MetricsRegistry, percentile
from askdocs.obs.tracing import Tracer
from askdocs.pipeline import RAGPipeline

log = logging.getLogger(__name__)


# (key, header, format). Order here is the column order of the table.
COLUMNS = [
    ("recall@5", "Recall@5", "{:.3f}"),
    ("mrr@10", "MRR@10", "{:.3f}"),
    ("ndcg@10", "nDCG@10", "{:.3f}"),
    ("answer_accuracy", "Answer acc.", "{:.1%}"),
    ("citation_precision", "Citation prec.", "{:.1%}"),
    ("support_ratio", "Supported sent.", "{:.1%}"),
    ("unanswerable_abstain_rate", "Abstain (OOS)", "{:.1%}"),
    ("false_abstain_rate", "False abstain", "{:.1%}"),
    ("latency_p50_ms", "p50 ms", "{:.1f}"),
    ("latency_p95_ms", "p95 ms", "{:.1f}"),
    ("cost_per_1k_usd", "$ / 1k q", "{:.4f}"),
]


def load_golden(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_variant(
    name: str, config_path: str | None, overrides: dict | list | None, golden: list[dict], judge=None, warmup: int = 3
) -> tuple[dict, list[dict]]:
    settings = load_settings(config_path, overrides)
    settings.name = name
    settings.observability.trace_dir = None  # eval traces stay in memory
    index = load_or_build(settings)
    by_id = {c.chunk_id: c for c in index.chunks}
    pipe = RAGPipeline(settings, index=index, tracer=Tracer([]), metrics=MetricsRegistry())

    for item in golden[:warmup]:  # load models / warm caches so p95 isn't one cold start
        pipe.answer(item["question"])

    rows: list[dict] = []
    for item in golden:
        a = pipe.answer(item["question"])
        ranked = [by_id[i] for i in a.quality.get("ranked", [])]
        cited = [by_id[c["chunk_id"]] for c in a.citations]
        ev = item["evidence"]
        n_rel = sum(1 for c in index.chunks if any(chunk_matches(c, e) for e in ev))
        row: dict[str, Any] = {
            "variant": name,
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "answer": a.text,
            "abstained": a.abstained,
            "abstain_reason": a.abstain_reason,
            "evidence_score": a.quality.get("evidence_score"),
            "latency_ms": a.timings_ms.get("ask", 0.0),
            "stages_ms": a.timings_ms,
            "cost_usd": a.usage["cost_usd"],
            "prompt_tokens": a.usage["prompt_tokens"],
            "completion_tokens": a.usage["completion_tokens"],
            "citation_retries": a.quality.get("citation_retries", 0),
            "support_ratio": a.quality.get("support_ratio"),
            "top5": [f"{c.doc_id} > {c.section}" for c in ranked[:5]],
        }
        if item["answerable"]:
            row.update(
                recall5=recall_at_k(ranked, ev, 5),
                mrr10=mrr_at_k(ranked, ev, 10),
                ndcg10=ndcg_at_k(ranked, ev, 10, n_rel),
                keyword_recall=0.0 if a.abstained else keyword_recall(a.text, item["must_include"]),
                citation_precision=None if a.abstained else citation_precision(cited, ev),
            )
            row["correct"] = (not a.abstained) and row["keyword_recall"] == 1.0
            if judge is not None and not a.abstained:
                row["judge"] = judge.grade(item["question"], a.text, item["answer"], a.contexts)
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    oos = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["abstained"]]
    lat = [r["latency_ms"] for r in rows]
    stage_names = sorted({k for r in rows for k in r["stages_ms"] if k != "ask"})
    metrics = {
        "n": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(oos),
        "recall@5": _mean([r["recall5"] for r in ans]),
        "mrr@10": _mean([r["mrr10"] for r in ans]),
        "ndcg@10": _mean([r["ndcg10"] for r in ans]),
        "answer_accuracy": _mean([1.0 if r["correct"] else 0.0 for r in ans]),
        "keyword_recall": _mean([r["keyword_recall"] for r in ans]),
        "citation_precision": _mean([r["citation_precision"] for r in answered if r["citation_precision"] is not None]),
        "support_ratio": _mean([r["support_ratio"] for r in rows if not r["abstained"] and r["support_ratio"] is not None]),
        "unanswerable_abstain_rate": _mean([1.0 if r["abstained"] else 0.0 for r in oos]),
        "false_abstain_rate": _mean([1.0 if r["abstained"] else 0.0 for r in ans]),
        "citation_retry_rate": _mean([1.0 if r["citation_retries"] else 0.0 for r in rows]),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "latency_p99_ms": percentile(lat, 99),
        "cost_per_1k_usd": 1000 * _mean([r["cost_usd"] for r in rows]),
        "avg_prompt_tokens": _mean([r["prompt_tokens"] for r in rows]),
        "stages_p50_ms": {s: percentile([r["stages_ms"][s] for r in rows if s in r["stages_ms"]], 50) for s in stage_names},
        "stages_p95_ms": {s: percentile([r["stages_ms"][s] for r in rows if s in r["stages_ms"]], 95) for s in stage_names},
    }
    judged = [r["judge"] for r in answered if r.get("judge") and r["judge"]["faithful"] is not None]
    if judged:
        metrics["judge_faithfulness"] = _mean([j["faithful"] for j in judged])
        metrics["judge_correctness"] = _mean([j["correct"] for j in judged])
    info = {
        "settings": settings.model_dump(include={"retrieval", "reranker", "embedding", "llm", "chunking", "citations", "gate"}),
        "index": index.manifest,
        "metrics": metrics,
    }
    return info, rows


def results_table(results: dict) -> str:
    primary = results.get("primary")
    head = "| Variant | " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    sep = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [head, sep]
    for name, v in results["variants"].items():
        m = v["metrics"]
        cells = [fmt.format(m[key]) for key, _, fmt in COLUMNS]
        label = f"**{name}** (gated)" if name == primary else name
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_markdown(results: dict) -> str:
    v = results["variants"].get(results.get("primary")) or next(iter(results["variants"].values()))
    idx = v["index"]
    stages = v["metrics"]["stages_p50_ms"]
    stage_line = ", ".join(
        f"{k} {ms:.2f}"
        for k, ms in sorted(stages.items(), key=lambda kv: -kv[1])
        if "." in k or k in ("rerank", "fuse", "generate", "gate")
    )
    return (
        f"{results_table(results)}\n\n"
        f"<sub>{results['n_questions']} questions ({v['metrics']['n_unanswerable']} out of scope), "
        f"{idx['n_chunks']} chunks from {idx['n_docs']} docs · profile `{results['config']}` · "
        f"embedding `{idx.get('embedding')}` · generator `{v['settings']['llm']['model']}` · "
        f"commit `{results.get('git_sha') or 'local'}` · {results['generated_at']}<br>"
        f"p50 per stage (ms), primary variant: {stage_line}</sub>"
    )


def update_readme(readme: Path, markdown: str, section: str = "CI") -> bool:
    start, end = f"<!-- RESULTS:{section}:START -->", f"<!-- RESULTS:{section}:END -->"
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        log.warning("README has no %s / %s markers; not updating", start, end)
        return False
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    readme.write_text(f"{pre}{start}\n{markdown}\n{end}{post}", encoding="utf-8")
    return True


def run_suite(
    config: str, suite: str | None, golden_path: str, out_dir: str, judge_config: str | None = None, only: list[str] | None = None
) -> dict:
    golden = load_golden(golden_path)
    if suite:
        spec = yaml.safe_load(Path(suite).read_text())
        variants = spec["variants"]
        primary = spec.get("primary", variants[0]["name"])
    else:
        variants, primary = [{"name": "default", "set": {}}], "default"
    if only:
        variants = [v for v in variants if v["name"] in only]

    judge = None
    if judge_config:
        from askdocs.config import LLMConfig
        from askdocs.eval.judge import Judge

        judge = Judge(LLMConfig.model_validate(yaml.safe_load(Path(judge_config).read_text())))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "git_sha": _git_sha(),
        "config": config,
        "golden": golden_path,
        "n_questions": len(golden),
        "primary": primary,
        "variants": {},
    }
    all_rows: list[dict] = []
    for v in variants:
        t0 = time.perf_counter()
        info, rows = evaluate_variant(v["name"], config, v.get("set") or {}, golden, judge=judge if v["name"] == primary else None)
        results["variants"][v["name"]] = info
        all_rows += rows
        m = info["metrics"]
        log.info(
            "%-28s recall@5=%.3f acc=%.3f p95=%.1fms (%.1fs)",
            v["name"],
            m["recall@5"],
            m["answer_accuracy"],
            m["latency_p95_ms"],
            time.perf_counter() - t0,
        )

    (out / "latest.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "latest.md").write_text(results_markdown(results) + "\n")
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return results


def calibrate_gate(config: str, golden_path: str, overrides: list[str] | None = None, step: float = 0.025) -> dict:
    """Sweep the evidence-gate threshold to show the trade-off between refusing
    out-of-scope questions and wrongly refusing answerable ones.

    It runs the golden set once with the gate off, then replays each threshold
    offline. Calibrate on a dev split if you can: tuning on the same questions
    you report makes the reported numbers optimistic."""
    from askdocs.config import apply_overrides

    golden = load_golden(golden_path)
    ov = apply_overrides({}, overrides or [])
    ov = apply_overrides(ov, {"gate.min_evidence_score": 0.0})
    flat: dict[str, Any] = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(v, f"{prefix}{k}.")
            else:
                flat[f"{prefix}{k}"] = v

    _flatten(ov)
    _, rows = evaluate_variant("calibration", config, flat, golden, warmup=0)
    ans = [r for r in rows if r["answerable"]]
    oos = [r for r in rows if not r["answerable"]]
    grid = []
    t = 0.0
    while t <= 1.0 + 1e-9:

        def gated(r: dict, thr: float = t) -> bool:
            return r["abstained"] or (r["evidence_score"] or 0.0) < thr

        oos_rate = _mean([1.0 if gated(r) else 0.0 for r in oos])
        false_rate = _mean([1.0 if gated(r) else 0.0 for r in ans])
        acc = _mean([1.0 if (r["correct"] and not gated(r)) else 0.0 for r in ans])
        grid.append(
            {
                "threshold": round(t, 3),
                "oos_abstain_rate": oos_rate,
                "false_abstain_rate": false_rate,
                "answer_accuracy": acc,
                "j": oos_rate - false_rate,
            }
        )
        t += step
    best = max(grid, key=lambda g: (round(g["j"], 6), -g["threshold"]))
    return {"grid": grid, "best": best, "n_answerable": len(ans), "n_unanswerable": len(oos)}
