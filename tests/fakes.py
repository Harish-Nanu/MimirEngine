"""In-memory stand-ins for MinIO, Qdrant and Kafka."""

from __future__ import annotations

import math

from qdrant_client import QdrantClient

from mimir.errors import TransientError
from mimir.status import QdrantStatusStore
from mimir.vectorstore import Chunk, SearchHit


def memory_status(settings) -> QdrantStatusStore:
    """The real status store on qdrant-client's in-process mode."""
    store = QdrantStatusStore(settings, client=QdrantClient(location=":memory:"))
    store.ensure_collection()
    return store


class FakeObjects:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put(self, bucket, key, stream, content_type):
        self.objects[(bucket, key)] = stream.read()

    def get(self, bucket, key):
        return self.objects[(bucket, key)]

    def delete(self, bucket, key):
        self.objects.pop((bucket, key), None)

    def ping(self):
        pass


class FakeVectors:
    def __init__(self):
        self.points: dict[str, Chunk] = {}

    def replace_document(self, doc_id, chunks):
        self.points = {pid: c for pid, c in self.points.items() if c.doc_id != doc_id}
        for c in chunks:
            self.points[c.point_id] = c

    def delete_document(self, doc_id):
        self.replace_document(doc_id, [])

    def ping(self):
        pass

    def search(self, vector, top_k, doc_id=None):
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b)) / (
                (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1
            )

        cands = [c for c in self.points.values() if doc_id is None or c.doc_id == doc_id]
        ranked = sorted(cands, key=lambda c: cos(vector, c.vector), reverse=True)[:top_k]
        return [SearchHit(cos(vector, c.vector), c.doc_id, c.index, c.text, c.metadata) for c in ranked]


class FakePublisher:
    def __init__(self, fail: bool = False):
        self.sent: list[tuple[str, bytes, dict]] = []
        self.published = []
        self.fail = fail

    def send(self, topic, value, key, failures, headers=None):
        self.sent.append((topic, value, headers or {}))

    def flush(self, failures, timeout=30.0):
        if self.fail:
            raise TransientError("broker down")

    def ping(self):
        if self.fail:
            raise TransientError("broker down")

    def publish(self, jobs):
        if self.fail:
            raise TransientError("broker down")
        self.published.extend(jobs)


class FakeMessage:
    def __init__(self, value: bytes, topic="mimir.ingest", partition=0, offset=0):
        self._value, self._topic, self._partition, self._offset = value, topic, partition, offset

    def value(self):
        return self._value

    def topic(self):
        return self._topic

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def error(self):
        return None


class FakeConsumer:
    def __init__(self):
        self.committed, self.seeks, self.paused = [], [], []

    def commit(self, message, asynchronous):
        self.committed.append(message.offset())

    def seek(self, tp):
        self.seeks.append((tp.topic, tp.partition, tp.offset))

    def pause(self, tps):
        self.paused.extend((tp.topic, tp.partition) for tp in tps)
