"""Dense retrieval: pluggable embedders plus a brute-force cosine index.

Brute force with numpy is the right call below roughly 100k chunks. It's exact,
has no build step, and a 5k x 384 matrix multiply takes well under a
millisecond. Swap in FAISS or a vector DB when the corpus grows past that.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
import requests

from askdocs.config import EmbeddingConfig


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class LSAEmbedder:
    """TF-IDF + truncated SVD (latent semantic analysis). It's not a neural
    embedding. It exists so CI and the tests run with zero downloads. The
    'local' and 'cloud' profiles use a real embedding model."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.name = f"lsa-{cfg.dims}"
        self._vec = None
        self._svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from askdocs.text import tokenize

        self._vec = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        x = self._vec.fit_transform(texts)
        n_comp = max(2, min(self.cfg.dims, x.shape[0] - 1, x.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self._svd.fit(x)

    def _transform(self, texts: list[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("LSAEmbedder.fit() must be called before embedding")
        return _l2(self._svd.transform(self._vec.transform(texts)))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]

    def __getstate__(self):  # picklable for the on-disk index
        return self.__dict__


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer  # optional dependency: pip install .[ml]

        self.cfg = cfg
        self.name = cfg.model
        self._model = SentenceTransformer(cfg.model)

    def fit(self, texts: list[str]) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [self.cfg.doc_prefix + t for t in texts]
        return _l2(self._model.encode(texts, batch_size=self.cfg.batch_size, normalize_embeddings=True, show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._model.encode([self.cfg.query_prefix + text], normalize_embeddings=True))[0]

    def __getstate__(self):
        raise TypeError("SentenceTransformerEmbedder is rebuilt from config, not pickled")


class _HTTPEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.name = cfg.model
        self.session = requests.Session()

    def fit(self, texts: list[str]) -> None:
        pass

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.cfg.batch_size):
            out.extend(self._call(texts[i : i + self.cfg.batch_size]))
        return _l2(np.array(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.cfg.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.cfg.query_prefix + text])[0]

    def _call(self, batch: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def __getstate__(self):
        raise TypeError("HTTP embedders are rebuilt from config, not pickled")


class OllamaEmbedder(_HTTPEmbedder):
    """Uses Ollama's /api/embed endpoint, e.g. model nomic-embed-text."""

    def _call(self, batch: list[str]) -> list[list[float]]:
        base = (self.cfg.base_url or "http://localhost:11434").rstrip("/").removesuffix("/v1")
        r = self.session.post(f"{base}/api/embed", json={"model": self.cfg.model, "input": batch}, timeout=120)
        r.raise_for_status()
        return r.json()["embeddings"]


class OpenAIEmbedder(_HTTPEmbedder):
    """Any OpenAI-compatible /embeddings endpoint."""

    def _call(self, batch: list[str]) -> list[list[float]]:
        base = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get(self.cfg.api_key_env or "OPENAI_API_KEY", "")
        r = self.session.post(
            f"{base}/embeddings",
            json={"model": self.cfg.model, "input": batch},
            headers={"Authorization": f"Bearer {key}"},
            timeout=120,
        )
        r.raise_for_status()
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


def make_embedder(cfg: EmbeddingConfig) -> Embedder:
    return {
        "lsa": LSAEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "ollama": OllamaEmbedder,
        "openai": OpenAIEmbedder,
    }[cfg.provider](cfg)


class DenseIndex:
    def __init__(self, embedder: Embedder, vectors: np.ndarray):
        self.embedder = embedder
        self.vectors = _l2(vectors)

    @classmethod
    def build(cls, embedder: Embedder, texts: list[str]) -> DenseIndex:
        embedder.fit(texts)
        return cls(embedder, embedder.embed_documents(texts))

    def search_vector(self, qv: np.ndarray, k: int) -> list[tuple[int, float]]:
        sims = self.vectors @ qv
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in top]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.search_vector(self.embedder.embed_query(query), k)

    def save(self, directory: Path) -> None:
        np.save(directory / "dense.npy", self.vectors)
        if isinstance(self.embedder, LSAEmbedder):
            with open(directory / "lsa.pkl", "wb") as f:
                pickle.dump(self.embedder, f)

    @classmethod
    def load(cls, directory: Path, cfg: EmbeddingConfig) -> DenseIndex:
        vectors = np.load(directory / "dense.npy")
        if cfg.provider == "lsa":
            # Only ever unpickle files this tool wrote into its own index dir.
            with open(directory / "lsa.pkl", "rb") as f:
                embedder = pickle.load(f)
        else:
            embedder = make_embedder(cfg)
        return cls(embedder, vectors)
