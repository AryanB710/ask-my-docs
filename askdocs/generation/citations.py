"""Citation enforcement.

Prompting for citations gets you citations. It doesn't get you correct ones.
After generation, every sentence is checked:

1. Does it cite at least one source, and are the cited numbers in range?
2. Is it supported? support = share of the sentence's content words found in
   the cited chunk(s). This is a lexical proxy for entailment: cheap,
   deterministic and explainable. It catches the common failure where the model
   cites [2] for a number that only appears in [4], or invents a number. It
   can't catch a sentence that reuses the source's words while reversing their
   meaning. For that, enable the LLM judge in eval (eval/judge.py).
3. Every number and identifier in the sentence (500, 99.99, LQ-1033, v2) must
   appear in a cited chunk. This is a hard rule, separate from the overlap score:
   a fabricated "9,999" in an otherwise copied sentence still scores 0.75 on overlap.

If the check fails, the pipeline regenerates once with the failures listed.
Then it drops the unsupported sentences, or abstains if nothing survives.
"""

from __future__ import annotations

import re

from askdocs.text import find_citations, split_answer_sentences, strip_citations, tokenize
from askdocs.types import CitationReport, Retrieved, SentenceCheck

_FILLER = {"answer", "source", "document", "according", "state", "note", "yes", "also", "however"}


def support_score(sentence: str, cited_texts: list[str]) -> float:
    s_toks = [t for t in tokenize(strip_citations(sentence)) if t not in _FILLER]
    if not s_toks:
        return 1.0  # nothing factual to check, e.g. "Yes [1]."
    pool: set[str] = set()
    for t in cited_texts:
        pool |= set(tokenize(t))
    # Numbers count double: a wrong number is the worst kind of hallucination in docs QA.
    weight = {t: (2.0 if re.search(r"\d", t) else 1.0) for t in s_toks}
    return sum(w for t, w in weight.items() if t in pool) / sum(weight.values())


def missing_numbers(sentence: str, cited_texts: list[str]) -> list[str]:
    pool: set[str] = set()
    for t in cited_texts:
        pool |= set(tokenize(t, stem=False, drop_stopwords=False))
    nums = [t for t in tokenize(strip_citations(sentence), stem=False, drop_stopwords=False) if re.search(r"\d", t)]
    return [t for t in dict.fromkeys(nums) if t not in pool]


def check_citations(answer: str, contexts: list[Retrieved], min_support: float) -> CitationReport:
    checks: list[SentenceCheck] = []
    n = len(contexts)
    for sent in split_answer_sentences(answer):
        cites = find_citations(sent)
        valid = [c for c in cites if 1 <= c <= n]
        invalid = [c for c in cites if not 1 <= c <= n]
        texts = [f"{contexts[c - 1].chunk.header}\n{contexts[c - 1].chunk.text}" for c in valid]
        sup = support_score(sent, texts) if valid else 0.0
        missing = missing_numbers(sent, texts) if valid else []
        ok = bool(valid) and sup >= min_support and not missing
        checks.append(SentenceCheck(sent, valid, invalid, round(sup, 3), ok, missing))
    return CitationReport(checks)


def problems_for_retry(report: CitationReport) -> list[str]:
    out = []
    for s in report.sentences:
        if not s.citations:
            out.append(f'"{s.sentence}" (no citation)')
        elif s.missing_numbers:
            out.append(f'"{s.sentence}" ({", ".join(s.missing_numbers)} does not appear in source {s.citations})')
        elif not s.supported:
            out.append(f'"{s.sentence}" (cited {s.citations} but the source does not state this)')
    return out


def keep_supported(report: CitationReport) -> str:
    return " ".join(s.sentence for s in report.sentences if s.supported)


def citation_payload(report: CitationReport, contexts: list[Retrieved], only_supported: bool = True) -> list[dict]:
    used = sorted({c for s in report.sentences if s.supported or not only_supported for c in s.citations})
    return [
        {
            "n": c,
            "chunk_id": contexts[c - 1].chunk.chunk_id,
            "doc_id": contexts[c - 1].chunk.doc_id,
            "header": contexts[c - 1].chunk.header,
            "snippet": contexts[c - 1].chunk.text[:280],
        }
        for c in used
    ]
