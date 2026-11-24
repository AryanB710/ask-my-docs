"""Optional LLM-as-judge for faithfulness and correctness.

The lexical support check in generation/citations.py is fast and deterministic,
but it can't tell "retries are allowed" from "retries are not allowed". When a
judge model is configured (askdocs eval --judge configs/judge.yaml), each
non-abstained answer is also graded by an LLM on:

  faithful: every claim is stated by the cited sources      (0 or 1)
  correct:  the answer agrees with the reference answer     (0 or 1)

It runs only in the nightly/full eval, never in the PR gate: it costs money and
judges are noisy. Spot-check the judge against your own labels before trusting it.
"""

from __future__ import annotations

import json
import re

from askdocs.config import LLMConfig
from askdocs.generation.llm import OpenAICompatibleLLM
from askdocs.generation.prompts import format_sources
from askdocs.types import Retrieved

JUDGE_PROMPT = """You are grading a documentation QA system. Reply with JSON only:
{{"faithful": 0 or 1, "correct": 0 or 1, "reason": "<one short sentence>"}}

faithful = 1 only if EVERY factual claim in the answer is directly stated in the sources.
correct  = 1 only if the answer agrees with the reference answer on the key facts.

Sources:
{sources}

Question: {question}
Reference answer: {reference}
System answer: {answer}"""


class Judge:
    def __init__(self, cfg: LLMConfig):
        self.llm = OpenAICompatibleLLM(cfg)

    def grade(self, question: str, answer: str, reference: str, contexts: list[Retrieved]) -> dict:
        prompt = JUDGE_PROMPT.format(sources=format_sources(contexts), question=question, reference=reference, answer=answer)
        # Reuse the HTTP client but send our own messages.
        payload_msgs = [{"role": "user", "content": prompt}]
        import requests

        r = self.llm.session.post(
            self.llm.cfg.base_url.rstrip("/") + "/chat/completions",
            json={"model": self.llm.cfg.model, "messages": payload_msgs, "temperature": 0, "max_tokens": 150},
            headers=self.llm.headers,
            timeout=self.llm.cfg.timeout_s,
        )
        try:
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            out = json.loads(m.group(0)) if m else {}
            return {"faithful": int(out.get("faithful", 0)), "correct": int(out.get("correct", 0)), "reason": out.get("reason", "")}
        except (requests.HTTPError, ValueError, KeyError, AttributeError) as e:
            return {"faithful": None, "correct": None, "reason": f"judge error: {e}"}
