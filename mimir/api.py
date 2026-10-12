"""HTTP API: upload documents, track their ingestion, search the index and ask Claude about it."""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .embeddings import Embedder
from .errors import FatalError, RateLimitError, TransientError
from .extract import is_supported
from .messages import IngestJob, doc_id_for_object
from .status import DELETED, FAILED, QUEUED, STATUSES, DocumentStatus

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


@dataclass
class Services:
    settings: Settings
    objects: object  # MinioStore-like: put(), delete(), ping()
    publisher: object  # JobPublisher-like: publish(), ping()
    embedder: Embedder
    vectors: object  # QdrantStore-like: search(), delete_document(), ping()
    status: object  # QdrantStatusStore-like
    answerer: object | None = None  # ClaudeAnswerer/OllamaAnswerer-like; None when not configured


def default_services() -> Services:
    from .answer import build_answerer
    from .embeddings import build_embedder
    from .queue import JobPublisher
    from .status import QdrantStatusStore
    from .storage import MinioStore
    from .vectorstore import QdrantStore

    settings = get_settings()
    try:
        answerer = build_answerer(settings)
    except FatalError as exc:
        log.warning("POST /ask disabled: %s", exc)
        answerer = None
    return Services(
        settings=settings,
        objects=MinioStore(settings),
        publisher=JobPublisher(settings),
        embedder=build_embedder(settings),
        vectors=QdrantStore(settings),
        status=QdrantStatusStore(settings),
        answerer=answerer,
    )


class UploadedDocument(BaseModel):
    doc_id: str
    filename: str
    key: str


class DocumentList(BaseModel):
    documents: list[DocumentStatus]
    next_cursor: str | None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    doc_id: str | None = None


class QueryHit(BaseModel):
    score: float
    doc_id: str
    chunk_index: int
    text: str
    metadata: dict


class QueryResponse(BaseModel):
    results: list[QueryHit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=8, ge=1, le=50)
    doc_id: str | None = None


class CitationOut(BaseModel):
    doc_id: str
    chunk_index: int
    filename: str
    cited_text: str
    ref: int | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    sources: list[QueryHit]
    model: str


def services(request: Request) -> Services:
    return request.app.state.services


def require_api_key(request: Request) -> None:
    expected = request.app.state.services.settings.api_key
    if not expected:
        return
    supplied = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        supplied = supplied or auth[7:]
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key")


def create_app(services_factory: Callable[[], Services] = default_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services_factory()
        yield

    app = FastAPI(title="MimirEngine", lifespan=lifespan)
    auth = [Depends(require_api_key)]

    @app.exception_handler(RateLimitError)
    async def rate_limited(request, exc: RateLimitError):
        headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else None
        return JSONResponse({"detail": str(exc)}, status.HTTP_429_TOO_MANY_REQUESTS, headers=headers)

    @app.exception_handler(TransientError)
    async def unavailable(request, exc: TransientError):
        return JSONResponse({"detail": str(exc)}, status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.exception_handler(FatalError)
    async def misconfigured(request, exc: FatalError):
        log.error("misconfiguration: %s", exc)
        return JSONResponse({"detail": f"server misconfigured: {exc}"}, status.HTTP_500_INTERNAL_SERVER_ERROR)

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(svc: Services = Depends(services)):
        """Readiness: every backing service answers."""
        checks = {}
        for name, dep in (("minio", svc.objects), ("kafka", svc.publisher),
                          ("qdrant", svc.vectors), ("status", svc.status)):
            try:
                dep.ping()
                checks[name] = "ok"
            except Exception as exc:
                checks[name] = f"error: {exc}"
        if svc.answerer is None:
            checks["llm"] = "not configured"
        else:
            try:
                svc.answerer.ping()
                checks["llm"] = "ok"
            except Exception as exc:
                checks["llm"] = f"error: {exc}"
        ok = all(v == "ok" for k, v in checks.items() if k != "llm")
        return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks},
                            status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=list[UploadedDocument],
              dependencies=auth)
    def upload(files: list[UploadFile], svc: Services = Depends(services)):
        limit = svc.settings.max_upload_mb * 1024 * 1024
        names = []
        for f in files:
            filename = os.path.basename((f.filename or "").replace("\\", "/")) or "unnamed"
            if f.size is not None and f.size > limit:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    f"{filename} exceeds {svc.settings.max_upload_mb} MB")
            if not is_supported(filename, f.content_type):
                raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                    f"unsupported file type: {filename}")
            names.append(filename)

        bucket = svc.settings.s3_bucket
        jobs, uploaded = [], []
        for f, filename in zip(files, names):
            key = f"uploads/{uuid.uuid4().hex}/{filename}"
            # Same derivation as `mimir enqueue`, so re-queuing an upload keeps its doc_id.
            doc_id = doc_id_for_object(bucket, key)
            svc.objects.put(bucket, key, f.file, f.content_type)
            jobs.append(IngestJob(doc_id=doc_id, bucket=bucket, key=key, filename=filename,
                                  content_type=f.content_type))
            uploaded.append(UploadedDocument(doc_id=doc_id, filename=filename, key=key))

        svc.status.put(*(DocumentStatus.for_job(j, QUEUED) for j in jobs))
        try:
            svc.publisher.publish(jobs)
        except TransientError as exc:
            # Files are stored; `mimir enqueue --prefix uploads/` re-queues them under the same doc_ids.
            try:
                svc.status.put(*(DocumentStatus.for_job(j, FAILED, error=f"queue unavailable: {exc}")
                                 for j in jobs))
            except TransientError:
                pass
            raise TransientError(f"queue unavailable: {exc}") from exc
        return uploaded

    @app.get("/documents", response_model=DocumentList, dependencies=auth)
    def list_documents(
        status_filter: str | None = Query(None, alias="status", enum=list(STATUSES)),
        limit: int = Query(50, ge=1, le=500),
        cursor: str | None = None,
        svc: Services = Depends(services),
    ):
        docs, next_cursor = svc.status.list(status_filter, limit, cursor)
        return DocumentList(documents=docs, next_cursor=next_cursor)

    @app.get("/documents/{doc_id}", response_model=DocumentStatus, dependencies=auth)
    def get_document(doc_id: str, svc: Services = Depends(services)):
        doc = _valid_doc_id(doc_id) and svc.status.get(doc_id)
        if not doc or doc.status == DELETED:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
        return doc

    @app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=auth)
    def delete_document(doc_id: str, svc: Services = Depends(services)):
        doc = _valid_doc_id(doc_id) and svc.status.get(doc_id)
        if not doc or doc.status == DELETED:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
        # Tombstone first, so a worker mid-job drops its result instead of resurrecting the document.
        doc.status, doc.error = DELETED, None
        svc.status.put(doc)
        svc.vectors.delete_document(doc_id)
        svc.objects.delete(doc.bucket, doc.key)

    @app.get("/stats", dependencies=auth)
    def stats(svc: Services = Depends(services)):
        return {"documents": svc.status.counts()}

    @app.post("/query", response_model=QueryResponse, dependencies=auth)
    def query(req: QueryRequest, svc: Services = Depends(services)):
        return QueryResponse(results=_search(svc, req.query, req.top_k, req.doc_id))

    @app.post("/ask", response_model=AskResponse, dependencies=auth)
    def ask(req: AskRequest, svc: Services = Depends(services)):
        if svc.answerer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "answers are not configured: set ANTHROPIC_API_KEY, or MIMIR_LLM_PROVIDER=ollama, "
                                "and restart the API")
        hits = _search(svc, req.question, req.top_k, req.doc_id)
        answer = svc.answerer.answer(req.question, hits)
        return AskResponse(
            answer=answer.text,
            citations=[CitationOut(**c.__dict__) for c in answer.citations],
            sources=hits,
            model=answer.model,
        )

    return app


def _search(svc: Services, text: str, top_k: int, doc_id: str | None) -> list[QueryHit]:
    if doc_id is not None and not _valid_doc_id(doc_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "doc_id must be a UUID")
    vector = svc.embedder.embed_query(text)
    return [QueryHit(**h.__dict__) for h in svc.vectors.search(vector, top_k, doc_id)]


def _valid_doc_id(doc_id: str) -> bool:
    try:
        uuid.UUID(doc_id)
        return True
    except ValueError:
        return False
