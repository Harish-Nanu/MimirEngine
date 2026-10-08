"""Process one ingestion job: download -> extract -> chunk -> embed -> store.

No Kafka here; the worker decides what to do with the exceptions this raises.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Protocol

from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .errors import RateLimitError, TransientError
from .extract import extract_text
from .messages import IngestJob
from .vectorstore import Chunk

log = logging.getLogger(__name__)


class ObjectStore(Protocol):
    def get(self, bucket: str, key: str) -> bytes: ...


class VectorStore(Protocol):
    def replace_document(self, doc_id: str, chunks: list[Chunk]) -> None: ...


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        objects: ObjectStore,
        embedder: Embedder,
        vectors: VectorStore,
        sleep=time.sleep,
        inline_retries: int = 4,
    ):
        self.settings = settings
        self.objects = objects
        self.embedder = embedder
        self.vectors = vectors
        self.sleep = sleep
        self.inline_retries = inline_retries

    def process(self, job: IngestJob) -> int:
        data = self.objects.get(job.bucket, job.key)
        text = extract_text(data, job.filename, job.content_type)
        pieces = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)

        vectors: list[list[float]] = []
        bs = self.settings.embedding_batch_size
        for i in range(0, len(pieces), bs):
            vectors.extend(self._embed_with_backoff(pieces[i : i + bs]))

        meta = {"filename": job.filename, "bucket": job.bucket, "key": job.key}
        chunks = [
            Chunk(doc_id=job.doc_id, index=i, text=t, vector=v, metadata=meta)
            for i, (t, v) in enumerate(zip(pieces, vectors, strict=True))
        ]
        self.vectors.replace_document(job.doc_id, chunks)
        log.info("ingested doc=%s file=%s chunks=%d", job.doc_id, job.filename, len(chunks))
        return len(chunks)

    def _embed_with_backoff(self, texts: list[str]) -> list[list[float]]:
        """Short in-process retries for rate limits/blips. If they run out, the error
        propagates and the worker re-queues the whole job on the retry topic."""
        for attempt in range(self.inline_retries + 1):
            try:
                return self.embedder.embed_documents(texts)
            except TransientError as exc:
                if attempt == self.inline_retries:
                    raise
                if isinstance(exc, RateLimitError) and exc.retry_after:
                    delay = exc.retry_after
                else:
                    delay = min(30.0, 2**attempt) * (0.5 + random.random() / 2)
                log.warning("embedding retry %d in %.1fs: %s", attempt + 1, delay, exc)
                self.sleep(delay)
        raise AssertionError("unreachable")
