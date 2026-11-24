"""askdocs: ingest | ask | eval | gate | report | serve"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/offline.yaml"


def _settings(args):
    from askdocs.config import load_settings

    return load_settings(args.config, args.set)


def cmd_ingest(args) -> int:
    from askdocs.ingest.index import build_index, save_index

    s = _settings(args)
    if args.corpus:
        s.corpus_dir = args.corpus
    idx = build_index(s, with_dense=s.retrieval.mode != "bm25")
    out = Path(s.index_dir) / s.index_key()
    save_index(idx, out)
    print(json.dumps({**idx.manifest, "path": str(out)}, indent=2))
    return 0


def cmd_ask(args) -> int:
    from askdocs.pipeline import RAGPipeline

    pipe = RAGPipeline(_settings(args))
    a = pipe.answer(args.question)
    if args.json:
        print(json.dumps(a.to_dict(include_contexts=args.contexts), indent=2))
        return 0
    print(f"\n{a.text}\n")
    for c in a.citations:
        print(f"  [{c['n']}] {c['doc_id']} > {c['header']}")
    stages = {k: v for k, v in a.timings_ms.items() if k != "ask"}
    print(f"\n  {a.timings_ms.get('ask', 0):.1f} ms total · " + " · ".join(f"{k} {v:.1f}" for k, v in stages.items()))
    print(f"  evidence {a.quality.get('evidence_score')} · cost ${a.usage['cost_usd']:.6f} · trace {a.trace_id}")
    if a.abstained:
        print(f"  abstained: {a.abstain_reason}")
    return 0


def cmd_eval(args) -> int:
    from askdocs.eval.runner import results_markdown, run_suite, update_readme

    results = run_suite(args.config, args.suite, args.golden, args.out, judge_config=args.judge, only=args.only)
    md = results_markdown(results)
    print(md)
    if args.update_readme:
        update_readme(Path(args.readme), md, args.readme_section)
        print(f"\nupdated {args.readme}")
    return 0


def cmd_calibrate(args) -> int:
    from askdocs.eval.runner import calibrate_gate

    res = calibrate_gate(args.config, args.golden, args.set, step=args.step)
    print(f"{'threshold':>9} {'OOS abstain':>12} {'false abstain':>14} {'answer acc':>11}")
    for g in res["grid"]:
        mark = "  <- best (max OOS abstain - false abstain)" if g is res["best"] else ""
        print(f"{g['threshold']:>9.3f} {g['oos_abstain_rate']:>12.1%} {g['false_abstain_rate']:>14.1%} {g['answer_accuracy']:>11.1%}{mark}")
    print(
        f"\n{res['n_unanswerable']} out-of-scope vs {res['n_answerable']} answerable questions. Small samples: treat this as a guide, not a fit."
    )
    return 0


def cmd_gate(args) -> int:
    from askdocs.eval.gate import run_gate

    return run_gate(args.results, args.thresholds, args.baseline, update_baseline=args.update_baseline)


def cmd_report(args) -> int:
    from askdocs.obs.dashboard import load_traces, summarize, write_dashboard

    traces = load_traces(args.traces)
    if not traces:
        print(f"no traces found in {args.traces}. Run some questions first (askdocs ask / the API).")
        return 1
    summary = summarize(traces)
    if args.html:
        write_dashboard(summary, traces, Path(args.html))
        print(f"wrote {args.html}")
    print(json.dumps({k: v for k, v in summary.items() if k != "recent"}, indent=2))
    return 0


def cmd_serve(args) -> int:
    import os

    import uvicorn

    os.environ["ASKDOCS_CONFIG"] = args.config
    if args.set:
        os.environ["ASKDOCS_SET"] = json.dumps(args.set)
    uvicorn.run("askdocs.api.server:app", host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="askdocs", description="Production-style RAG over your docs, with evals and tracing.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", "-c", default=DEFAULT_CONFIG)
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field, e.g. retrieval.mode=bm25")

    sp = sub.add_parser("ingest", help="chunk + index a folder of docs")
    common(sp)
    sp.add_argument("corpus", nargs="?", help="folder of .md/.txt/.html/.pdf (default: corpus_dir from config)")
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("ask", help="ask one question")
    common(sp)
    sp.add_argument("question")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--contexts", action="store_true", help="include retrieved chunks in --json output")
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("eval", help="run the golden set (optionally an ablation suite)")
    sp.add_argument("--config", "-c", default=DEFAULT_CONFIG)
    sp.add_argument("--suite", default=None, help="e.g. eval/suites/ablation.yaml")
    sp.add_argument("--only", nargs="*", help="run only these variant names")
    sp.add_argument("--golden", default="eval/golden.jsonl")
    sp.add_argument("--out", default="eval/results")
    sp.add_argument("--judge", default=None, help="LLM judge config (adds faithfulness/correctness)")
    sp.add_argument("--update-readme", action="store_true")
    sp.add_argument("--readme", default="README.md")
    sp.add_argument("--readme-section", default="CI", help="which <!-- RESULTS:<NAME>:START --> block to replace")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("calibrate", help="sweep the abstention-gate threshold")
    common(sp)
    sp.add_argument("--golden", default="eval/golden.jsonl")
    sp.add_argument("--step", type=float, default=0.025)
    sp.set_defaults(fn=cmd_calibrate)

    sp = sub.add_parser("gate", help="fail (exit 1) if eval results regress")
    sp.add_argument("--results", default="eval/results/latest.json")
    sp.add_argument("--thresholds", default="eval/thresholds.yaml")
    sp.add_argument("--baseline", default="eval/baseline.json")
    sp.add_argument("--update-baseline", action="store_true")
    sp.set_defaults(fn=cmd_gate)

    sp = sub.add_parser("report", help="latency/cost/quality summary from traces")
    sp.add_argument("--traces", default="runs/traces")
    sp.add_argument("--html", default=None, help="also write a static HTML dashboard here")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("serve", help="run the HTTP API + web UI")
    common(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.cmd == "eval" else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
