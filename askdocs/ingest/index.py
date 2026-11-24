"""Builds and caches the index (chunks + BM25 + dense vectors).

The cache key is a hash of the corpus files plus the chunking/embedding config.
Changing chunk size or switching embedding model triggers a rebuild, so you
never query vectors that were built with different settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from askdocs.config import Settings
from askdocs.ingest.chunker import chunk_corpus
from askdocs.ingest.loaders import SUPPORTED, load_corpus
from askdocs.retrieval.bm25 import BM25Index
from askdocs.retrieval.dense import DenseIndex, make_embedder
from askdocs.types import Chunk

log = logging.getLogger(__name__)


def corpus_fingerprint(corpus_dir: str | Path) -> str:
    h = hashlib.sha256()
    root = Path(corpus_dir)
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@dataclass
class Index:
    chunks: list[Chunk]
    bm25: BM25Index
    dense: DenseIndex | None
    manifest: dict

    @property
    def size(self) -> int:
        return len(self.chunks)


def _texts(chunks: list[Chunk], settings: Settings) -> list[str]:
    return [c.index_text(settings.chunking.contextual_headers) for c in chunks]


def build_index(settings: Settings, with_dense: bool = True) -> Index:
    t0 = time.perf_counter()
    docs = load_corpus(settings.corpus_dir)
    chunks = chunk_corpus(docs, settings.chunking)
    texts = _texts(chunks, settings)
    bm25 = BM25Index(texts, settings.retrieval.bm25_k1, settings.retrieval.bm25_b)
    dense = DenseIndex.build(make_embedder(settings.embedding), texts) if with_dense else None
    manifest = {
        "index_key": settings.index_key(),
        "corpus_fingerprint": corpus_fingerprint(settings.corpus_dir),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "embedding": settings.embedding.model if with_dense else None,
        "build_seconds": round(time.perf_counter() - t0, 3),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return Index(chunks, bm25, dense, manifest)


def save_index(index: Index, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in index.chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    if index.dense is not None:
        index.dense.save(directory)
    (directory / "manifest.json").write_text(json.dumps(index.manifest, indent=2))


def load_or_build(settings: Settings, persist: bool = True) -> Index:
    needs_dense = settings.retrieval.mode in ("dense", "hybrid")
    directory = Path(settings.index_dir) / settings.index_key()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        fresh = manifest.get("corpus_fingerprint") == corpus_fingerprint(settings.corpus_dir)
        has_dense = (directory / "dense.npy").exists()
        if fresh and (has_dense or not needs_dense):
            chunks = [Chunk(**json.loads(line)) for line in (directory / "chunks.jsonl").open(encoding="utf-8")]
            bm25 = BM25Index(_texts(chunks, settings), settings.retrieval.bm25_k1, settings.retrieval.bm25_b)
            dense = DenseIndex.load(directory, settings.embedding) if needs_dense else None
            log.info("loaded index %s (%d chunks)", directory, len(chunks))
            return Index(chunks, bm25, dense, manifest)
        log.info("index at %s is stale; rebuilding", directory)
    index = build_index(settings, with_dense=needs_dense)
    if persist:
        save_index(index, directory)
    return index
