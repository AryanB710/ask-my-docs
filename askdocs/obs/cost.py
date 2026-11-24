"""Cost per request from token usage and a pricing table in configs/pricing.yaml.

Prices change often, so they live in config rather than code. Unknown models
cost 0 and log a warning once, so a missing entry is visible and doesn't crash."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
_warned: set[str] = set()


@lru_cache(maxsize=8)
def load_pricing(path: str) -> dict[str, dict[str, float]]:
    p = Path(path)
    if not p.exists():
        return {}
    return (yaml.safe_load(p.read_text()) or {}).get("models", {})


def estimate_tokens(text: str) -> int:
    """About 4 characters per token for English. Used only when the provider
    doesn't return usage (the extractive generator, some local servers)."""
    return max(1, len(text) // 4) if text else 0


def request_cost(model: str, prompt_tokens: int, completion_tokens: int, pricing_file: str) -> float:
    table = load_pricing(pricing_file)
    entry = table.get(model)
    if entry is None:
        if model not in _warned and model != "extractive":
            log.warning("no pricing entry for model %r in %s; counting cost as 0", model, pricing_file)
            _warned.add(model)
        return 0.0
    return (prompt_tokens * entry.get("input_per_1m", 0.0) + completion_tokens * entry.get("output_per_1m", 0.0)) / 1_000_000
