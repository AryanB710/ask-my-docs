"""Generators.

- OpenAICompatibleLLM: one client for OpenAI, Groq, Together, vLLM, LM Studio and
  Ollama (http://localhost:11434/v1), because they all speak /chat/completions.
- ExtractiveGenerator: deterministic, model-free. Picks the best-matching
  sentences from the top chunks and cites them. CI uses it so the eval gate
  runs on every PR with no API key and no flakiness, and it gives a floor that
  any real LLM should beat.
"""

from __future__ import annotations

import os
import random
import time
from typing import Protocol

import requests

from askdocs.config import LLMConfig
from askdocs.generation.prompts import ABSTAIN_TOKEN, build_messages
from askdocs.obs.cost import estimate_tokens
from askdocs.retrieval.bm25 import BM25Index
from askdocs.text import split_sentences, tokenize
from askdocs.types import Generation, Retrieved


class Generator(Protocol):
    model: str

    def generate(self, question: str, contexts: list[Retrieved], problems: list[str] | None = None) -> Generation: ...


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.model = cfg.model
        self.session = requests.Session()
        key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = f"Bearer {key}"

    def generate(self, question: str, contexts: list[Retrieved], problems: list[str] | None = None) -> Generation:
        payload = {
            "model": self.cfg.model,
            "messages": build_messages(question, contexts, problems),
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.session.post(url, json=payload, headers=self.headers, timeout=self.cfg.timeout_s)
                if r.status_code in self.RETRY_STATUS and attempt < self.cfg.max_retries:
                    retry_after = r.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                    raise LLMError(f"HTTP {r.status_code}", delay)
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                text = (data["choices"][0]["message"].get("content") or "").strip()
                return Generation(
                    text=text,
                    model=self.cfg.model,
                    prompt_tokens=int(usage.get("prompt_tokens") or estimate_tokens(str(payload["messages"]))),
                    completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
                    raw={"finish_reason": data["choices"][0].get("finish_reason")},
                )
            except (requests.ConnectionError, requests.Timeout, LLMError) as e:
                last_err = e
                if attempt >= self.cfg.max_retries:
                    break
                hinted = e.args[1] if isinstance(e, LLMError) and len(e.args) > 1 else None
                # exponential backoff with full jitter, capped at 8s
                time.sleep(hinted if hinted is not None else random.uniform(0, min(8.0, 0.5 * 2**attempt)))
        raise LLMError(f"LLM call failed after {self.cfg.max_retries + 1} attempts: {last_err}")


class ExtractiveGenerator:
    model = "extractive"

    def __init__(self, bm25: BM25Index, max_sentences: int = 2, min_score: float = 0.25):
        self.bm25 = bm25
        self.max_sentences = max_sentences
        self.min_score = min_score

    def _score(self, q_weights: dict[str, float], total: float, unit: str) -> float:
        toks = set(tokenize(unit))
        return sum(w for t, w in q_weights.items() if t in toks) / total if total else 0.0

    def generate(self, question: str, contexts: list[Retrieved], problems: list[str] | None = None) -> Generation:
        q = list(dict.fromkeys(tokenize(question)))
        weights = {t: self.bm25.term_idf(t) for t in q}
        total = sum(weights.values())
        scored: list[tuple[float, int, str]] = []
        for n, r in enumerate(contexts, start=1):
            header_toks = set(tokenize(r.chunk.header))
            header_bonus = sum(w for t, w in weights.items() if t in header_toks) / total if total else 0.0
            for unit in split_sentences(r.chunk.text):
                if unit.startswith("```") or unit.startswith("| HTTP") or len(unit.split()) < 3:
                    continue
                s = self._score(weights, total, unit) + 0.15 * header_bonus + 0.02 * (len(contexts) - n)
                scored.append((s, n, unit))
        scored.sort(key=lambda x: -x[0])
        prompt_tokens = estimate_tokens(str(build_messages(question, contexts)))
        if not scored or scored[0][0] < self.min_score:
            return Generation(ABSTAIN_TOKEN, self.model, prompt_tokens, 1)
        best = scored[0][0]
        picked = [x for x in scored if x[0] >= 0.75 * best][: self.max_sentences]
        parts = []
        for _, n, unit in picked:
            unit = unit.strip().strip("|").strip()
            unit = " - ".join(c.strip() for c in unit.split("|")) if "|" in unit else unit
            unit = unit.rstrip(".:;,") + f" [{n}]."
            parts.append(unit)
        text = " ".join(parts)
        return Generation(text, self.model, prompt_tokens, estimate_tokens(text))


def make_generator(cfg: LLMConfig, bm25: BM25Index) -> Generator:
    if cfg.provider == "extractive":
        return ExtractiveGenerator(bm25)
    return OpenAICompatibleLLM(cfg)
