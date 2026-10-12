"""Per-document ingestion status, kept as vectorless points in a Qdrant collection.

Lifecycle: queued -> processing -> ready, or -> retrying -> ... -> failed.
Deleting a document leaves a "deleted" tombstone so in-flight jobs for it are skipped.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .config import Settings
from .errors import TransientError
from .messages import IngestJob

QUEUED, PROCESSING, READY, RETRYING, FAILED, DELETED = (
    "queued", "processing", "ready", "retrying", "failed", "deleted",
)
STATUSES = (QUEUED, PROCESSING, READY, RETRYING, FAILED, DELETED)


@dataclass
class DocumentStatus:
    doc_id: str
    filename: str
    bucket: str
    key: str
    status: str
    attempt: int = 0
    chunks: int | None = None
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def for_job(cls, job: IngestJob, status: str, **kw) -> DocumentStatus:
        now = time.time()
        return cls(doc_id=job.doc_id, filename=job.filename, bucket=job.bucket, key=job.key,
                   status=status, attempt=job.attempt, created_at=now, updated_at=now, **kw)


class QdrantStatusStore:
    def __init__(self, settings: Settings, client=None):
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.client = client
        self.collection = settings.documents_collection

    def ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(self.collection, vectors_config={})
        for field in ("status", "bucket"):
            self.client.create_payload_index(
                self.collection, field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        self.client.create_payload_index(
            self.collection, "updated_at", field_schema=models.PayloadSchemaType.FLOAT
        )

    def get(self, doc_id: str) -> DocumentStatus | None:
        try:
            records = self.client.retrieve(self.collection, [doc_id], with_payload=True)
        except Exception as exc:
            raise TransientError(f"status lookup failed: {exc}") from exc
        return DocumentStatus(**records[0].payload) if records else None

    def put(self, *records: DocumentStatus) -> None:
        from qdrant_client import models

        if not records:
            return
        try:
            existing = {
                r.id: r.payload
                for r in self.client.retrieve(self.collection, [r.doc_id for r in records], with_payload=True)
            }
            points = []
            for rec in records:
                old = existing.get(rec.doc_id)
                if old:
                    rec.created_at = old.get("created_at") or rec.created_at
                rec.updated_at = time.time()
                points.append(models.PointStruct(id=rec.doc_id, vector={}, payload=asdict(rec)))
            self.client.upsert(self.collection, points=points, wait=True)
        except Exception as exc:
            raise TransientError(f"status write failed: {exc}") from exc

    def list(self, status: str | None = None, limit: int = 50, cursor: str | None = None
             ) -> tuple[list[DocumentStatus], str | None]:
        from qdrant_client import models

        if status:
            flt = models.Filter(must=[_match("status", status)])
        else:
            flt = models.Filter(must_not=[_match("status", DELETED)])
        try:
            records, next_id = self.client.scroll(
                self.collection, scroll_filter=flt, limit=limit, offset=cursor, with_payload=True
            )
        except Exception as exc:
            raise TransientError(f"status listing failed: {exc}") from exc
        return [DocumentStatus(**r.payload) for r in records], (str(next_id) if next_id else None)

    def counts(self) -> dict[str, int]:
        from qdrant_client import models

        try:
            return {
                s: self.client.count(self.collection, count_filter=models.Filter(must=[_match("status", s)]),
                                     exact=True).count
                for s in STATUSES
            }
        except Exception as exc:
            raise TransientError(f"status count failed: {exc}") from exc

    def ping(self) -> None:
        self.client.collection_exists(self.collection)


def _match(key: str, value: str):
    from qdrant_client import models

    return models.FieldCondition(key=key, match=models.MatchValue(value=value))
