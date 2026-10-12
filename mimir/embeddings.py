"""Pluggable embedding providers.

Every provider maps provider-specific failures onto mimir.errors so the pipeline can
decide uniformly: RateLimitError/TransientError -> back off and retry,
PermanentError -> DLQ, FatalError -> stop the worker.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from abc import ABC, abstractmethod

import httpx

from .config import Settings
from .errors import FatalError, PermanentError, RateLimitError, TransientError


class Embedder(ABC):
    _dimension: int | None = None

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension


class RateLimiter:
    """Spaces calls at least 1/max_rps apart, shared across threads in one process."""

    def __init__(self, max_rps: float):
        self.interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing. Lexical only; for tests and offline dev, not production."""

    def __init__(self, dimension: int = 256):
        self._dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for token in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vec[h % self._dimension] += 1.0 if (h >> 63) else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class LocalEmbedder(Embedder):
    """sentence-transformers model on CPU/GPU. No API key and no remote rate limits."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}

    def __init__(self, model: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise FatalError("local embeddings need `pip install mimir-engine[local]`") from exc
        self.model_name = model or self.DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        prefix = self.QUERY_PREFIX.get(self.model_name, "")
        return self.model.encode([prefix + text], normalize_embeddings=True)[0].tolist()


class _HTTPEmbedder(Embedder):
    url: str

    def __init__(self, api_key: str | None, model: str, limiter: RateLimiter, timeout: float = 60.0):
        if not api_key:
            raise FatalError(f"{type(self).__name__} requires MIMIR_EMBEDDING_API_KEY")
        self.model = model
        self.limiter = limiter
        self.client = httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    def _post(self, payload: dict) -> list[list[float]]:
        self.limiter.wait()
        try:
            resp = self.client.post(self.url, json=payload)
        except httpx.TransportError as exc:
            raise TransientError(f"embedding request failed: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError("embedding provider rate limit", _retry_after(resp))
        if resp.status_code in (401, 403):
            raise FatalError(f"embedding provider rejected credentials ({resp.status_code})")
        if resp.status_code >= 500 or resp.status_code == 408:
            raise TransientError(f"embedding provider error {resp.status_code}")
        if resp.status_code >= 400:
            raise PermanentError(f"embedding request rejected {resp.status_code}: {resp.text[:300]}")

        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


class OpenAIEmbedder(_HTTPEmbedder):
    url = "https://api.openai.com/v1/embeddings"
    DEFAULT_MODEL = "text-embedding-3-small"

    def embed_documents(self, texts):
        return self._post({"model": self.model, "input": texts})

    def embed_query(self, text):
        return self._post({"model": self.model, "input": [text]})[0]


class VoyageEmbedder(_HTTPEmbedder):
    url = "https://api.voyageai.com/v1/embeddings"
    DEFAULT_MODEL = "voyage-3.5"

    def embed_documents(self, texts):
        return self._post({"model": self.model, "input": texts, "input_type": "document"})

    def embed_query(self, text):
        return self._post({"model": self.model, "input": [text], "input_type": "query"})[0]


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    limiter = RateLimiter(settings.embedding_max_rps)
    if provider == "local":
        return LocalEmbedder(settings.embedding_model)
    if provider == "openai":
        return OpenAIEmbedder(
            settings.embedding_api_key, settings.embedding_model or OpenAIEmbedder.DEFAULT_MODEL, limiter
        )
    if provider == "voyage":
        return VoyageEmbedder(
            settings.embedding_api_key, settings.embedding_model or VoyageEmbedder.DEFAULT_MODEL, limiter
        )
    if provider == "hash":
        return HashEmbedder()
    raise FatalError(f"unknown embedding provider {settings.embedding_provider!r}")
