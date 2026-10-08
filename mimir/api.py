"""HTTP API: upload documents for ingestion and query the vector index."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .embeddings import Embedder
from .errors import TransientError
from .messages import IngestJob


@dataclass
class Services:
    settings: Settings
    objects: object  # MinioStore-like: put()
    publisher: object  # JobPublisher-like: publish()
    embedder: Embedder
    vectors: object  # QdrantStore-like: search()


def default_services() -> Services:
    from .embeddings import build_embedder
    from .queue import JobPublisher
    from .storage import MinioStore
    from .vectorstore import QdrantStore

    settings = get_settings()
    return Services(
        settings=settings,
        objects=MinioStore(settings),
        publisher=JobPublisher(settings),
        embedder=build_embedder(settings),
        vectors=QdrantStore(settings),
    )


class UploadedDocument(BaseModel):
    doc_id: str
    filename: str
    key: str


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


def create_app(services_factory: Callable[[], Services] = default_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services_factory()
        yield

    app = FastAPI(title="MimirEngine", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=list[UploadedDocument])
    def upload(files: list[UploadFile], request: Request):
        svc: Services = request.app.state.services
        bucket = svc.settings.s3_bucket
        jobs, uploaded = [], []
        for f in files:
            filename = os.path.basename(f.filename or "") or "unnamed"
            doc_id = str(uuid.uuid4())
            key = f"uploads/{doc_id}/{filename}"
            svc.objects.put(bucket, key, f.file, f.content_type)
            jobs.append(IngestJob(doc_id=doc_id, bucket=bucket, key=key, filename=filename,
                                  content_type=f.content_type))
            uploaded.append(UploadedDocument(doc_id=doc_id, filename=filename, key=key))
        try:
            svc.publisher.publish(jobs)
        except TransientError as exc:
            # Files are stored; `mimir enqueue` can re-queue them once Kafka is back.
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"queue unavailable: {exc}")
        return uploaded

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, request: Request):
        svc: Services = request.app.state.services
        vector = svc.embedder.embed_query(req.query)
        hits = svc.vectors.search(vector, req.top_k, req.doc_id)
        return QueryResponse(results=[QueryHit(**h.__dict__) for h in hits])

    return app
