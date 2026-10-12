"""Qdrant vector storage for document chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .config import Settings
from .errors import FatalError, TransientError

UPSERT_BATCH = 256


@dataclass
class Chunk:
    doc_id: str
    index: int
    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        # Deterministic, so a redelivered job overwrites instead of duplicating.
        return str(uuid.uuid5(uuid.UUID(self.doc_id), str(self.index)))


@dataclass
class SearchHit:
    score: float
    doc_id: str
    chunk_index: int
    text: str
    metadata: dict


class QdrantStore:
    def __init__(self, settings: Settings):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.collection = settings.qdrant_collection

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors.size
            if existing != dimension:
                raise FatalError(
                    f"collection {self.collection!r} has dimension {existing}, embedder produces "
                    f"{dimension}; use a new MIMIR_QDRANT_COLLECTION when switching models"
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        self.client.create_payload_index(
            self.collection, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD
        )

    def replace_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Remove any previous chunks for doc_id, then write the new ones."""
        from qdrant_client import models

        try:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
                wait=True,
            )
            for i in range(0, len(chunks), UPSERT_BATCH):
                batch = chunks[i : i + UPSERT_BATCH]
                self.client.upsert(
                    self.collection,
                    points=[
                        models.PointStruct(
                            id=c.point_id,
                            vector=c.vector,
                            payload={"doc_id": c.doc_id, "chunk_index": c.index, "text": c.text, **c.metadata},
                        )
                        for c in batch
                    ],
                    wait=True,
                )
        except Exception as exc:
            raise _classify(exc) from exc

    def delete_document(self, doc_id: str) -> None:
        from qdrant_client import models

        try:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
                wait=True,
            )
        except Exception as exc:
            raise _classify(exc) from exc

    def search(self, vector: list[float], top_k: int, doc_id: str | None = None) -> list[SearchHit]:
        try:
            result = self.client.query_points(
                self.collection,
                query=vector,
                limit=top_k,
                query_filter=_doc_filter(doc_id) if doc_id else None,
                with_payload=True,
            )
        except Exception as exc:
            raise _classify(exc) from exc
        hits = []
        for p in result.points:
            payload = dict(p.payload or {})
            hits.append(
                SearchHit(
                    score=p.score,
                    doc_id=payload.pop("doc_id", ""),
                    chunk_index=payload.pop("chunk_index", -1),
                    text=payload.pop("text", ""),
                    metadata=payload,
                )
            )
        return hits

    def ping(self) -> None:
        self.client.collection_exists(self.collection)


def _doc_filter(doc_id: str):
    from qdrant_client import models

    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def _classify(exc: Exception) -> Exception:
    from qdrant_client.http.exceptions import UnexpectedResponse

    if isinstance(exc, UnexpectedResponse) and exc.status_code in (400, 404):
        # Wrong vector size or missing collection: configuration, not this document.
        return FatalError(f"qdrant rejected request ({exc.status_code}): {exc.content[:300]!r}")
    return TransientError(f"qdrant request failed: {exc}")
