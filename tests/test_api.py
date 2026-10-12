"""Document lifecycle, status tracking, auth, limits and error mapping."""

import uuid

from mimir.embeddings import HashEmbedder
from mimir.errors import RateLimitError, TransientError
from mimir.pipeline import Pipeline
from mimir.status import DELETED, DocumentStatus
from mimir.worker import Worker

from .fakes import FakeConsumer, FakeMessage, FakeObjects, FakePublisher, FakeVectors, memory_status
from .test_pipeline import SETTINGS, FlakyEmbedder, api_client, make_job


def ingest(client, name="a.txt", body=b"Kafka moves messages between services."):
    """Upload through the API, then run the queued job through a real worker."""
    svc = client.app.state.services
    (doc,) = client.post("/documents", files=[("files", (name, body, "text/plain"))]).json()
    job = svc.publisher.published[-1]
    pipeline = Pipeline(svc.settings, svc.objects, svc.embedder, svc.vectors, sleep=lambda s: None)
    worker = Worker(svc.settings, pipeline, FakePublisher(), FakeConsumer(), status=svc.status)
    worker.handle(FakeMessage(job.to_bytes()))
    return doc["doc_id"]


# --- status lifecycle -------------------------------------------------------


def test_document_lifecycle_upload_ready_delete():
    with api_client()[0] as client:
        svc = client.app.state.services
        doc_id = ingest(client)

        doc = client.get(f"/documents/{doc_id}").json()
        assert doc["status"] == "ready" and doc["chunks"] == 1 and doc["filename"] == "a.txt"
        assert doc["created_at"] <= doc["updated_at"]
        assert [d["doc_id"] for d in client.get("/documents?status=ready").json()["documents"]] == [doc_id]
        assert client.get("/stats").json()["documents"]["ready"] == 1

        assert client.delete(f"/documents/{doc_id}").status_code == 204
        assert not svc.vectors.points and not svc.objects.objects
        assert client.get(f"/documents/{doc_id}").status_code == 404
        assert client.get("/documents").json()["documents"] == []
        assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_unknown_or_malformed_doc_id_is_404():
    with api_client()[0] as client:
        assert client.get(f"/documents/{uuid.uuid4()}").status_code == 404
        assert client.get("/documents/not-a-uuid").status_code == 404
        assert client.post("/query", json={"query": "x", "doc_id": "nope"}).status_code == 422


def test_worker_records_retrying_and_failed():
    status = memory_status(SETTINGS)
    pipeline = Pipeline(SETTINGS, _objects(), FlakyEmbedder([TransientError("down")] * 100), _vectors(),
                        sleep=lambda s: None)
    worker = Worker(SETTINGS, pipeline, FakePublisher(), FakeConsumer(), status=status)

    worker.handle(FakeMessage(make_job().to_bytes()))
    rec = status.get(make_job().doc_id)
    assert rec.status == "retrying" and rec.attempt == 1 and "down" in rec.error

    worker.handle(FakeMessage(make_job(attempt=SETTINGS.max_attempts - 1).to_bytes()))
    assert status.get(make_job().doc_id).status == "failed"


def test_worker_skips_deleted_document():
    status = memory_status(SETTINGS)
    status.put(DocumentStatus.for_job(make_job(), DELETED))
    vectors, consumer = _vectors(), FakeConsumer()
    worker = Worker(SETTINGS, Pipeline(SETTINGS, _objects(), _embedder(), vectors), FakePublisher(), consumer,
                    status=status)
    worker.handle(FakeMessage(make_job().to_bytes(), offset=3))
    assert consumer.committed == [3] and not vectors.points
    assert status.get(make_job().doc_id).status == DELETED


def test_worker_drops_chunks_if_deleted_mid_job():
    status = memory_status(SETTINGS)
    vectors = _vectors()

    class DeletingPipeline(Pipeline):
        def process(self, job):
            n = super().process(job)
            status.put(DocumentStatus.for_job(job, DELETED))  # API delete lands mid-job
            return n

    worker = Worker(SETTINGS, DeletingPipeline(SETTINGS, _objects(), _embedder(), vectors), FakePublisher(),
                    FakeConsumer(), status=status)
    worker.handle(FakeMessage(make_job().to_bytes()))
    assert not vectors.points
    assert status.get(make_job().doc_id).status == DELETED


# --- auth, limits, readiness, error mapping ---------------------------------


def test_api_key_required_when_configured():
    settings = SETTINGS.model_copy(update={"api_key": "s3cret"})
    with api_client(settings=settings)[0] as client:
        assert client.get("/health").status_code == 200
        assert client.get("/documents").status_code == 401
        assert client.get("/documents", headers={"x-api-key": "wrong"}).status_code == 401
        assert client.get("/documents", headers={"x-api-key": "s3cret"}).status_code == 200
        assert client.get("/documents", headers={"authorization": "Bearer s3cret"}).status_code == 200


def test_upload_rejects_oversize_and_unsupported_before_storing():
    settings = SETTINGS.model_copy(update={"max_upload_mb": 0})
    client, svc = api_client(settings=settings)
    with client:
        r = client.post("/documents", files=[("files", ("a.txt", b"alpha", "text/plain"))])
        assert r.status_code == 413
    with api_client()[0] as client:
        r = client.post("/documents", files=[("files", ("ok.txt", b"a", "text/plain")),
                                             ("files", ("x.exe", b"MZ", "application/octet-stream"))])
        assert r.status_code == 415
        assert not client.app.state.services.objects.objects  # nothing half-uploaded


def test_ready_reports_each_dependency():
    with api_client()[0] as client:
        r = client.get("/ready")
        assert r.status_code == 200 and r.json()["checks"]["llm"] == "not configured"
    with api_client(FakePublisher(fail=True))[0] as client:
        r = client.get("/ready")
        assert r.status_code == 503 and r.json()["checks"]["kafka"].startswith("error")


def test_query_maps_backend_errors():
    with api_client()[0] as client:
        svc = client.app.state.services

        def rate_limited(text):
            raise RateLimitError("slow down", retry_after=12)

        svc.embedder.embed_query = rate_limited
        r = client.post("/query", json={"query": "x"})
        assert r.status_code == 429 and r.headers["retry-after"] == "12"

        def down(*a, **k):
            raise TransientError("qdrant down")

        svc.embedder.embed_query = lambda text: [0.0] * 256
        svc.vectors.search = down
        assert client.post("/query", json={"query": "x"}).status_code == 503


def test_ui_is_served():
    with api_client()[0] as client:
        r = client.get("/")
        assert r.status_code == 200 and "MimirEngine" in r.text



def _objects():
    o = FakeObjects()
    o.objects[("raw", "k/doc.txt")] = b"Qdrant stores vectors."
    return o


def _vectors():
    return FakeVectors()


def _embedder():
    return HashEmbedder()
