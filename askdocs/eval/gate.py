"""CI regression gate.

Two kinds of checks on the primary variant:
  1. Absolute floors and ceilings from eval/thresholds.yaml. They catch
     "this is simply not good enough".
  2. Regressions against eval/baseline.json, the last accepted run on main.
     They catch "this PR made it worse", even when it is still above the floor.

Exit code 1 if anything fails. The markdown report goes to stdout and, inside
GitHub Actions, to the job summary.

To accept a deliberate change (new corpus, new model), run
`askdocs gate --update-baseline` and commit eval/baseline.json in the same PR.
That way baseline changes are reviewed like code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

HIGHER_IS_BETTER = {
    "recall@5",
    "mrr@10",
    "ndcg@10",
    "answer_accuracy",
    "keyword_recall",
    "citation_precision",
    "support_ratio",
    "unanswerable_abstain_rate",
    "judge_faithfulness",
    "judge_correctness",
}
LOWER_IS_BETTER = {"false_abstain_rate", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "cost_per_1k_usd", "citation_retry_rate"}


@dataclass
class Check:
    metric: str
    kind: str
    current: float
    reference: float | None
    limit: str
    passed: bool


def evaluate_gate(results: dict, thresholds: dict, baseline: dict | None) -> list[Check]:
    variant = thresholds.get("variant") or results["primary"]
    if variant not in results["variants"]:
        raise SystemExit(f"gate: variant {variant!r} not found in results (have: {list(results['variants'])})")
    cur = results["variants"][variant]["metrics"]
    checks: list[Check] = []

    for metric, rule in (thresholds.get("absolute") or {}).items():
        if metric not in cur:
            continue
        v = cur[metric]
        if "min" in rule:
            checks.append(Check(metric, "floor", v, None, f">= {rule['min']}", v >= rule["min"]))
        if "max" in rule:
            checks.append(Check(metric, "ceiling", v, None, f"<= {rule['max']}", v <= rule["max"]))

    reg = thresholds.get("regression") or {}
    if baseline and reg:
        base = baseline["variants"].get(variant, {}).get("metrics", {})
        max_drop = reg.get("max_quality_drop", 0.02)
        lat_slack = reg.get("max_latency_increase_pct", 50) / 100
        cost_slack = reg.get("max_cost_increase_pct", 10) / 100
        for metric in reg.get("metrics", sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)):
            if metric not in cur or metric not in base:
                continue
            v, b = cur[metric], base[metric]
            if metric in HIGHER_IS_BETTER:
                checks.append(Check(metric, "regression", v, b, f">= {b - max_drop:.3f}", v >= b - max_drop - 1e-9))
            elif metric.startswith("latency"):
                # small absolute allowance too, so a 0.4ms -> 0.9ms blip on a noisy runner doesn't fail
                lim = b * (1 + lat_slack) + reg.get("latency_floor_ms", 5)
                checks.append(Check(metric, "regression", v, b, f"<= {lim:.1f}", v <= lim))
            elif metric == "cost_per_1k_usd":
                lim = b * (1 + cost_slack) + 1e-6
                checks.append(Check(metric, "regression", v, b, f"<= {lim:.4f}", v <= lim))
            else:
                checks.append(Check(metric, "regression", v, b, f"<= {b + max_drop:.3f}", v <= b + max_drop + 1e-9))
    return checks


def _fmt(metric: str, v: float | None) -> str:
    if v is None:
        return "–"
    if metric.endswith("_ms"):
        return f"{v:.1f}"
    if metric.endswith("_usd"):
        return f"{v:.4f}"
    return f"{v:.3f}"


def render(checks: list[Check], variant: str) -> str:
    failed = [c for c in checks if not c.passed]
    title = "✅ Eval gate passed" if not failed else f"❌ Eval gate failed ({len(failed)} check{'s' if len(failed) != 1 else ''})"
    lines = [
        f"### {title}",
        f"Variant: `{variant}`",
        "",
        "| Metric | Check | Baseline | Current | Δ | Limit | |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for c in sorted(checks, key=lambda c: (c.passed, c.metric)):
        delta = "" if c.reference is None else _fmt(c.metric, c.current - c.reference)
        if delta and not delta.startswith("-"):
            delta = "+" + delta
        lines.append(
            f"| {c.metric} | {c.kind} | {_fmt(c.metric, c.reference)} | {_fmt(c.metric, c.current)} | {delta} | {c.limit} | {'✅' if c.passed else '❌'} |"
        )
    return "\n".join(lines)


def run_gate(results_path: str, thresholds_path: str, baseline_path: str, update_baseline: bool = False) -> int:
    results = json.loads(Path(results_path).read_text())
    thresholds = yaml.safe_load(Path(thresholds_path).read_text()) or {}
    variant = thresholds.get("variant") or results["primary"]
    if update_baseline:
        Path(baseline_path).write_text(json.dumps(results, indent=2))
        print(f"baseline updated from {results_path} -> {baseline_path}")
        return 0
    bp = Path(baseline_path)
    baseline = json.loads(bp.read_text()) if bp.exists() else None
    checks = evaluate_gate(results, thresholds, baseline)
    report = render(checks, variant)
    if baseline is None:
        report += "\n\n_No baseline found; only absolute thresholds were checked._"
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    Path(results_path).with_name("gate.md").write_text(report + "\n")
    return 0 if all(c.passed for c in checks) else 1
