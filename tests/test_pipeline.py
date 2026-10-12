import json
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from mimir.api import Services, create_app
from mimir.config import Settings
from mimir.embeddings import HashEmbedder, OpenAIEmbedder, RateLimiter
from mimir.errors import FatalError, PermanentError, RateLimitError, TransientError
from mimir.messages import IngestJob, doc_id_for_object
from mimir.pipeline import Pipeline
from mimir.worker import Worker, route_failure

from .fakes import FakeConsumer, FakeMessage, FakeObjects, FakePublisher, FakeVectors, memory_status

SETTINGS = Settings(_env_file=None, chunk_size=200, chunk_overlap=20, embedding_batch_size=4, max_attempts=3)
DOC = str(uuid.uuid4())


def make_job(**kw):
    defaults = dict(doc_id=DOC, bucket="raw", key="k/doc.txt", filename="doc.txt")
    return IngestJob(**{**defaults, **kw})


class FlakyEmbedder(HashEmbedder):
    def __init__(self, failures: list[Exception]):
        super().__init__()
        self.failures = failures

    def embed_documents(self, texts):
        if self.failures:
            raise self.failures.pop(0)
        return super().embed_documents(texts)


@pytest.fixture
def objects():
    o = FakeObjects()
    o.objects[("raw", "k/doc.txt")] = ("Kafka moves messages. " * 30 + "\n\n" + "MinIO stores objects. " * 30).encode()
    return o


# --- pipeline ---------------------------------------------------------------


def test_pipeline_stores_all_chunks_idempotently(objects):
    vectors = FakeVectors()
    p = Pipeline(SETTINGS, objects, HashEmbedder(), vectors)
    n = p.process(make_job())
    assert n > 4  # spans several embedding batches
    assert len(vectors.points) == n
    assert {c.metadata["filename"] for c in vectors.points.values()} == {"doc.txt"}

    # Redelivery overwrites instead of duplicating.
    p.process(make_job())
    assert len(vectors.points) == n

    # A shorter new version leaves no stale chunks behind.
    objects.objects[("raw", "k/doc.txt")] = b"tiny"
    assert p.process(make_job()) == 1
    assert len(vectors.points) == 1


def test_pipeline_backs_off_on_rate_limit(objects):
    sleeps = []
    embedder = FlakyEmbedder([RateLimitError(retry_after=7), TransientError("blip")])
    p = Pipeline(SETTINGS, objects, embedder, FakeVectors(), sleep=sleeps.append)
    p.process(make_job())
    assert sleeps[0] == 7  # honours Retry-After
    assert len(sleeps) == 2


def test_pipeline_gives_up_after_inline_retries(objects):
    embedder = FlakyEmbedder([RateLimitError()] * 10)
    p = Pipeline(SETTINGS, objects, embedder, FakeVectors(), sleep=lambda s: None, inline_retries=2)
    with pytest.raises(RateLimitError):
        p.process(make_job())
    assert len(embedder.failures) == 7  # 3 calls made


# --- routing ----------------------------------------------------------------


def test_route_failure():
    job = make_job()
    assert route_failure(job, PermanentError("bad"), SETTINGS).topic == SETTINGS.dlq_topic

    r = route_failure(job, TransientError("blip"), SETTINGS)
    assert r.topic == SETTINGS.retry_topic
    assert r.job.attempt == 1 and r.job.not_before > time.time()
    assert "blip" in r.job.last_error

    last = make_job(attempt=SETTINGS.max_attempts - 1)
    assert route_failure(last, TransientError("blip"), SETTINGS).topic == SETTINGS.dlq_topic


def make_worker(objects, embedder=None, publisher=None):
    publisher = publisher or FakePublisher()
    consumer = FakeConsumer()
    pipeline = Pipeline(SETTINGS, objects, embedder or HashEmbedder(), FakeVectors(), sleep=lambda s: None)
    return Worker(SETTINGS, pipeline, publisher, consumer), publisher, consumer


def test_worker_commits_success(objects):
    w, pub, con = make_worker(objects)
    w.handle(FakeMessage(make_job().to_bytes(), offset=5))
    assert con.committed == [5] and pub.sent == []


def test_worker_sends_corrupt_file_to_dlq(objects):
    objects.objects[("raw", "k/doc.txt")] = b"\x00\x00binary"
    w, pub, con = make_worker(objects)
    w.handle(FakeMessage(make_job().to_bytes(), offset=1))
    (topic, value, _), = pub.sent
    assert topic == SETTINGS.dlq_topic
    assert "PermanentError" in json.loads(value)["last_error"]
    assert con.committed == [1]


def test_worker_retries_then_dead_letters(objects):
    w, pub, con = make_worker(objects, embedder=FlakyEmbedder([TransientError("down")] * 100))
    w.handle(FakeMessage(make_job().to_bytes()))
    assert pub.sent[-1][0] == SETTINGS.retry_topic
    w.handle(FakeMessage(make_job(attempt=SETTINGS.max_attempts - 1).to_bytes()))
    assert pub.sent[-1][0] == SETTINGS.dlq_topic
    assert len(con.committed) == 2


def test_worker_defers_retry_that_is_not_due(objects):
    w, pub, con = make_worker(objects)
    msg = FakeMessage(make_job(attempt=1, not_before=time.time() + 60).to_bytes(),
                      topic=SETTINGS.retry_topic, partition=3, offset=9)
    w.handle(msg)
    assert con.committed == [] and pub.sent == []
    assert con.seeks == [(SETTINGS.retry_topic, 3, 9)]
    assert con.paused == [(SETTINGS.retry_topic, 3)]


def test_worker_stops_on_fatal_without_commit(objects):
    w, pub, con = make_worker(objects, embedder=FlakyEmbedder([FatalError("bad key")]))
    with pytest.raises(FatalError):
        w.handle(FakeMessage(make_job().to_bytes()))
    assert con.committed == [] and pub.sent == []


def test_worker_does_not_commit_if_dlq_unreachable(objects, monkeypatch):
    monkeypatch.setattr("mimir.worker.time.sleep", lambda s: None)
    objects.objects[("raw", "k/doc.txt")] = b""
    w, pub, con = make_worker(objects, publisher=FakePublisher(fail=True))
    w.handle(FakeMessage(make_job().to_bytes(), offset=4))
    assert con.committed == []
    assert con.seeks == [("mimir.ingest", 0, 4)]  # will be redelivered


def test_worker_dead_letters_malformed_message(objects):
    w, pub, con = make_worker(objects)
    w.handle(FakeMessage(b"garbage", offset=2))
    (topic, value, headers), = pub.sent
    assert topic == SETTINGS.dlq_topic and value == b"garbage" and "mimir-error" in headers
    assert con.committed == [2]


# --- embedding provider error mapping ---------------------------------------


def openai_with(handler):
    e = OpenAIEmbedder("sk-test", "text-embedding-3-small", RateLimiter(0))
    e.client = httpx.Client(transport=httpx.MockTransport(handler))
    return e


def test_openai_success_sorted_by_index():
    body = {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
    e = openai_with(lambda req: httpx.Response(200, json=body))
    assert e.embed_documents(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "status,headers,exc",
    [(429, {"retry-after": "3"}, RateLimitError), (503, {}, TransientError),
     (401, {}, FatalError), (400, {}, PermanentError)],
)
def test_openai_error_mapping(status, headers, exc):
    e = openai_with(lambda req: httpx.Response(status, headers=headers, json={}))
    with pytest.raises(exc) as info:
        e.embed_documents(["a"])
    if status == 429:
        assert info.value.retry_after == 3.0


def test_http_embedder_requires_key():
    with pytest.raises(FatalError):
        OpenAIEmbedder(None, "m", RateLimiter(0))


# --- API --------------------------------------------------------------------


def api_client(publisher=None, settings=SETTINGS, answerer=None):
    objects, vectors = FakeObjects(), FakeVectors()
    svc = Services(settings, objects, publisher or FakePublisher(), HashEmbedder(), vectors,
                   memory_status(settings), answerer)
    return TestClient(create_app(lambda: svc)), svc


def test_api_upload_stores_and_queues():
    with api_client()[0] as client:
        svc = client.app.state.services
        r = client.post("/documents", files=[("files", ("../../etc/a.txt", b"alpha", "text/plain")),
                                             ("files", ("b.md", b"beta", "text/markdown"))])
        assert r.status_code == 202
        docs = r.json()
        assert [d["filename"] for d in docs] == ["a.txt", "b.md"]  # path stripped
        assert len(svc.publisher.published) == 2
        assert svc.objects.objects[(SETTINGS.s3_bucket, docs[0]["key"])] == b"alpha"
        # Same doc_id that `mimir enqueue` would derive, so re-queuing never duplicates.
        assert docs[0]["doc_id"] == doc_id_for_object(SETTINGS.s3_bucket, docs[0]["key"])
        assert svc.status.get(docs[0]["doc_id"]).status == "queued"


def test_api_upload_reports_queue_outage():
    with api_client(FakePublisher(fail=True))[0] as client:
        r = client.post("/documents", files=[("files", ("a.txt", b"alpha", "text/plain"))])
        assert r.status_code == 503
        (doc,), _ = client.app.state.services.status.list()
        assert doc.status == "failed" and "queue unavailable" in doc.error


def test_api_query_end_to_end_with_pipeline():
    with api_client()[0] as client:
        svc = client.app.state.services
        svc.objects.objects[("raw", "k/doc.txt")] = b"Qdrant stores vectors.\n\nKafka streams events."
        Pipeline(SETTINGS, svc.objects, svc.embedder, svc.vectors).process(make_job())
        svc.objects.objects[("raw", "other.txt")] = b"Bananas are yellow fruit."
        Pipeline(SETTINGS, svc.objects, svc.embedder, svc.vectors).process(
            make_job(doc_id=str(uuid.uuid4()), key="other.txt", filename="other.txt"))

        r = client.post("/query", json={"query": "yellow bananas", "top_k": 1})
        assert r.status_code == 200
        (hit,) = r.json()["results"]
        assert "Bananas" in hit["text"] and hit["metadata"]["filename"] == "other.txt"

        r = client.post("/query", json={"query": "yellow bananas", "doc_id": DOC})
        assert all(h["doc_id"] == DOC for h in r.json()["results"])

        assert client.post("/query", json={"query": ""}).status_code == 422
