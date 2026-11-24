"""Tokenisation and sentence splitting shared by BM25, the lite reranker,
the extractive generator and the citation checker.

Using the same tokenizer everywhere matters: if BM25 and the citation
checker disagree on what counts as a word, the support scores drift.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does did for from had has have how i if in
    into is it its may me might my no not of on or our should so than that the their them then there
    these they this those to up was we were what when where which who why will with would you your
    after before about any all also each other per get got am""".split()
)

# Keep identifiers like lq-1033, retention_seconds, 99.99 and ap-south as one token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*")
_DIGIT_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3})")
_CITATION_RE = re.compile(r"\[(\d+)\]")


def normalize(text: str) -> str:
    """Lowercase and drop thousands separators so '3,000' matches '3000'."""
    return _DIGIT_COMMA_RE.sub("", text.lower())


def light_stem(tok: str) -> str:
    """A deliberately tiny stemmer: plurals and -ing/-ed. Porter is overkill here
    and mangles identifiers (e.g. 'retries' -> 'retri')."""
    if len(tok) <= 3 or not tok.isalpha():
        return tok
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            if suf == "s" and tok.endswith(("ss", "us", "is")):
                return tok
            if suf == "es" and not tok.endswith(("sses", "xes", "ches", "shes")):
                continue
            return tok[: -len(suf)] + rep
    return tok


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(normalize(text))
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if stem:
        toks = [light_stem(t) for t in toks]
    return toks


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def find_citations(text: str) -> list[int]:
    return [int(m) for m in _CITATION_RE.findall(text)]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`(\"'\[])|(?<=[.!?])(?=\[\d+\]\s)")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. Lines that are list items or table rows
    count as their own units, because that's how people write docs."""
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|"):
            if set(line) <= set("|-: "):  # markdown table separator row
                continue
            units.append(line)
            continue
        if line.startswith(("- ", "* ", "+ ")):
            line = line[2:].strip()
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`(\"'])", line) if s.strip())
    return units


def split_answer_sentences(text: str) -> list[str]:
    """Like split_sentences, but keeps trailing citation markers attached to the
    sentence they follow: 'A is 5 [1]. B is 6 [2].' -> ['A is 5 [1].', 'B is 6 [2].']"""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`(\"'])|(?<=\])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]
