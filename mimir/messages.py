"""Kafka message schema for ingestion jobs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace

from .errors import PermanentError

# Stable namespace so the same bucket/key always maps to the same doc_id.
DOC_NAMESPACE = uuid.UUID("6f1c2a4e-9d3b-4c8a-b0e5-2f7d9a1c3e58")


def doc_id_for_object(bucket: str, key: str) -> str:
    return str(uuid.uuid5(DOC_NAMESPACE, f"{bucket}/{key}"))


@dataclass(frozen=True)
class IngestJob:
    doc_id: str
    bucket: str
    key: str
    filename: str
    content_type: str | None = None
    attempt: int = 0
    not_before: float = 0.0  # unix time; retry-topic consumers wait until then
    enqueued_at: float = field(default_factory=time.time)
    last_error: str | None = None

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> IngestJob:
        try:
            data = json.loads(raw)
            return cls(**data)
        except (ValueError, TypeError) as exc:
            raise PermanentError(f"malformed job message: {exc}") from exc

    def next_attempt(self, error: str, delay_s: float) -> IngestJob:
        return replace(
            self,
            attempt=self.attempt + 1,
            not_before=time.time() + delay_s,
            last_error=error,
        )
