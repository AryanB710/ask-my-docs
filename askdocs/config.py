"""Typed config. One YAML file per profile (offline / local / cloud); any field can
be overridden from the CLI with --set retrieval.mode=bm25, which is how the
ablation suite builds its variants."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    max_words: int = 120
    overlap_words: int = 25
    contextual_headers: bool = True


class EmbeddingConfig(BaseModel):
    # lsa = TF-IDF + truncated SVD. It's an offline stand-in for CI, not a neural model.
    provider: Literal["lsa", "sentence-transformers", "ollama", "openai"] = "lsa"
    model: str = "lsa-256"
    dims: int = 256
    base_url: str | None = None
    api_key_env: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    batch_size: int = 32


class RerankConfig(BaseModel):
    provider: Literal["none", "lite", "cross-encoder"] = "lite"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 20


class RetrievalConfig(BaseModel):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    bm25_k: int = 30
    dense_k: int = 30
    rrf_k: int = 60
    final_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class LLMConfig(BaseModel):
    # extractive = deterministic sentence picker, no model. Lets CI run without API keys.
    provider: Literal["extractive", "openai-compatible"] = "extractive"
    model: str = "extractive"
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 400
    timeout_s: float = 60.0
    max_retries: int = 2


class CitationConfig(BaseModel):
    enforce: bool = True
    min_support: float = 0.6
    max_regenerations: int = 1
    on_failure: Literal["drop", "abstain"] = "drop"


class GateConfig(BaseModel):
    # Refuse before calling the LLM if the best chunk scores below this.
    # Saves cost on out-of-scope questions and stops "creative" answers.
    min_evidence_score: float = 0.30


class ObservabilityConfig(BaseModel):
    trace_dir: str | None = "runs/traces"
    pricing_file: str = "configs/pricing.yaml"
    otel: bool = False
    service_name: str = "ask-my-docs"


class Settings(BaseModel):
    name: str = "default"
    corpus_dir: str = "data/demo/corpus"
    index_dir: str = "runs/index"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankConfig = Field(default_factory=RerankConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    citations: CitationConfig = Field(default_factory=CitationConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def index_key(self) -> str:
        """Hash of everything that changes the index. If it differs from the stored
        manifest, the index is rebuilt instead of silently serving stale vectors."""
        payload = {
            "corpus_dir": self.corpus_dir,
            "chunking": self.chunking.model_dump(),
            "embedding": self.embedding.model_dump(exclude={"batch_size", "api_key_env"}),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(data: dict[str, Any], overrides: list[str] | dict[str, Any] | None) -> dict[str, Any]:
    data = copy.deepcopy(data)
    if not overrides:
        return data
    items = overrides.items() if isinstance(overrides, dict) else (o.split("=", 1) for o in overrides)
    for key, raw in items:
        value = _coerce(raw) if isinstance(raw, str) else raw
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return data


def load_settings(path: str | Path | None = None, overrides: list[str] | dict[str, Any] | None = None) -> Settings:
    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    return Settings.model_validate(apply_overrides(data, overrides))
