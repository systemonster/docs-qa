"""Pluggable embedders. Every embedder returns unit-length vectors of EMBED_DIM floats."""

import hashlib
import itertools
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from docs_qa.config import Settings

EMBED_DIM = 384  # fixed by the vector(384) column in schema.sql


class Embedder(Protocol):
    name: str  # part of the ingest fingerprint: changing it triggers re-embedding

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashEmbedder:
    """Offline lexical baseline: signed hashing of words and word pairs. Not semantic.

    Used by the tests and CI (no model download) and as a baseline in the eval.
    """

    name = "hash-384"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)

    @staticmethod
    def _one(text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * EMBED_DIM
        for feat in words + [f"{a} {b}" for a, b in itertools.pairwise(words)]:
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % EMBED_DIM
            vec[idx] += 1.0 if digest[4] & 1 else -1.0
        return _normalize(vec)


class FastEmbedder:
    """Local ONNX model through fastembed (default: BAAI/bge-small-en-v1.5, 384 dims)."""

    def __init__(self, model: str, cache_dir: str | None = None) -> None:
        self.name = f"fastembed:{model}"
        self._model_name = model
        self._cache_dir = cache_dir
        self._model = None  # loaded lazily so importing the app stays fast

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name, cache_dir=self._cache_dir)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        # bge models expect an instruction prefix on queries; query_embed adds it
        return next(iter(self._load().query_embed(text))).tolist()


class OpenAIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint, asked for 384 dims (text-embedding-3 can)."""

    def __init__(
        self, api_key: str | None, base_url: str, model: str, client: httpx.Client | None = None
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDER=openai")
        self.name = f"openai:{model}:{EMBED_DIM}"
        self._model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._client = client or httpx.Client(timeout=60)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = list(texts[i : i + 64])
            resp = self._client.post(
                self._url,
                headers=self._headers,
                json={"model": self._model, "input": batch, "dimensions": EMBED_DIM},
            )
            resp.raise_for_status()
            rows = sorted(resp.json()["data"], key=lambda r: r["index"])
            out.extend(_normalize(r["embedding"]) for r in rows)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def build_embedder(s: Settings) -> Embedder:
    if s.embedder == "hash":
        return HashEmbedder()
    if s.embedder == "openai":
        return OpenAIEmbedder(s.openai_api_key, s.openai_base_url, s.openai_embed_model)
    return FastEmbedder(s.fastembed_model, s.fastembed_cache_dir)
