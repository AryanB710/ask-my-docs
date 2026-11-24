"""Structure-aware chunking.

Rules, in order:
1. Never cross a heading boundary, because a chunk should belong to one section.
2. Inside a section, pack whole blocks (paragraphs, list items, table rows, code
   fences) up to max_words.
3. Only split a single block by sentences if it alone is larger than max_words.
4. Carry the tail of the previous chunk forward (overlap_words) when a section is
   split, so a fact that straddles the boundary is still retrievable.
"""

from __future__ import annotations

import hashlib
import re

from askdocs.config import ChunkingConfig
from askdocs.ingest.loaders import Document
from askdocs.text import split_sentences
from askdocs.types import Chunk


def _blocks(text: str) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            buf.append(line)
            in_code = not in_code
            if not in_code:
                blocks.append("\n".join(buf))
                buf = []
            continue
        if in_code:
            buf.append(line)
            continue
        is_unit_line = stripped.startswith(("|", "- ", "* ", "+ ")) or re.match(r"^\d+\.\s", stripped)
        if not stripped or is_unit_line:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            if is_unit_line:
                blocks.append(stripped)
            continue
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return [b.strip() for b in blocks if b.strip()]


def _wc(s: str) -> int:
    return len(s.split())


def _units(block: str, max_words: int) -> list[str]:
    if _wc(block) <= max_words or block.startswith("```"):
        return [block]
    return split_sentences(block) or [block]


def _join(units: list[str]) -> str:
    # One unit per line keeps table rows and list items intact and makes
    # sentence splitting downstream trivial.
    return "\n".join(units).strip()


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        section_name = " > ".join(section.path[1:]) if len(section.path) > 1 else section.path[0]
        units = [u for b in _blocks(section.text) for u in _units(b, cfg.max_words)]
        current: list[str] = []
        words = 0
        groups: list[list[str]] = []
        for u in units:
            w = _wc(u)
            if current and words + w > cfg.max_words:
                groups.append(current)
                tail: list[str] = []
                tw = 0
                for prev in reversed(current):
                    if tw + _wc(prev) > cfg.overlap_words:
                        break
                    tail.insert(0, prev)
                    tw += _wc(prev)
                current, words = tail[:], tw
            current.append(u)
            words += w
        if current:
            groups.append(current)
        for g in groups:
            text = _join(g)
            pos = len(chunks)
            digest = hashlib.sha1(f"{doc.doc_id}|{section_name}|{text}".encode()).hexdigest()[:8]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{pos}-{digest}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section_name,
                    text=text,
                    position=pos,
                    metadata={"source_path": doc.source_path},
                )
            )
    return chunks


def chunk_corpus(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, cfg)]
