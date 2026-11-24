from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def header(self) -> str:
        return f"{self.title} > {self.section}" if self.section and self.section != self.title else self.title

    def index_text(self, contextual_header: bool) -> str:
        """The text that is indexed. With contextual headers on, a chunk that just says
        'The default is 4 days.' becomes findable by 'retention period' too."""
        return f"{self.header}\n{self.text}" if contextual_header else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    source: str  # bm25 | dense | hybrid | rerank
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] | None = None


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[int]
    invalid_citations: list[int]
    support: float
    supported: bool
    missing_numbers: list[str] = field(default_factory=list)


@dataclass
class CitationReport:
    sentences: list[SentenceCheck]

    @property
    def coverage(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(1 for s in self.sentences if s.citations) / len(self.sentences)

    @property
    def support_ratio(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(1 for s in self.sentences if s.supported) / len(self.sentences)

    @property
    def ok(self) -> bool:
        return bool(self.sentences) and all(s.supported for s in self.sentences)


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    abstain_reason: str | None
    citations: list[dict[str, Any]]
    contexts: list[Retrieved]
    trace_id: str
    timings_ms: dict[str, float]
    usage: dict[str, Any]
    quality: dict[str, Any]

    def to_dict(self, include_contexts: bool = True) -> dict[str, Any]:
        d = {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "citations": self.citations,
            "trace_id": self.trace_id,
            "timings_ms": self.timings_ms,
            "usage": self.usage,
            "quality": self.quality,
        }
        if include_contexts:
            d["contexts"] = [
                {
                    "n": i + 1,
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "header": r.chunk.header,
                    "score": round(r.score, 4),
                    "ranks": r.ranks,
                    "text": r.chunk.text,
                }
                for i, r in enumerate(self.contexts)
            ]
        return d
