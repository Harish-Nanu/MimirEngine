"""Command line entry point: `mimir init | worker | api | enqueue`."""

from __future__ import annotations

import argparse
import logging
import os

from .config import get_settings
from .messages import IngestJob, doc_id_for_object


def cmd_init(args) -> None:
    """Create Kafka topics, the MinIO bucket and the Qdrant collection (idempotent)."""
    from .embeddings import build_embedder
    from .queue import ensure_topics
    from .storage import MinioStore
    from .vectorstore import QdrantStore

    settings = get_settings()
    ensure_topics(settings)
    MinioStore(settings).ensure_bucket(settings.s3_bucket)
    QdrantStore(settings).ensure_collection(build_embedder(settings).dimension)
    logging.info("init complete")


def cmd_worker(args) -> None:
    from .embeddings import build_embedder
    from .pipeline import Pipeline
    from .queue import JobPublisher
    from .storage import MinioStore
    from .vectorstore import QdrantStore
    from .worker import Worker

    settings = get_settings()
    embedder = build_embedder(settings)
    vectors = QdrantStore(settings)
    vectors.ensure_collection(embedder.dimension)
    pipeline = Pipeline(settings, MinioStore(settings), embedder, vectors)
    Worker(settings, pipeline, JobPublisher(settings)).run()


def cmd_api(args) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


def cmd_enqueue(args) -> None:
    """Queue every object under a bucket prefix. Doc ids derive from bucket/key, so re-running is safe."""
    from .queue import JobPublisher
    from .storage import MinioStore

    settings = get_settings()
    bucket = args.bucket or settings.s3_bucket
    publisher = JobPublisher(settings)
    batch: list[IngestJob] = []
    total = 0
    for key in MinioStore(settings).list_keys(bucket, args.prefix):
        batch.append(IngestJob(doc_id=doc_id_for_object(bucket, key), bucket=bucket, key=key,
                               filename=os.path.basename(key)))
        if len(batch) >= 500:
            publisher.publish(batch)
            total += len(batch)
            batch = []
    if batch:
        publisher.publish(batch)
        total += len(batch)
    logging.info("queued %d object(s) from s3://%s/%s", total, bucket, args.prefix)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=os.environ.get("MIMIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(prog="mimir")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create topics, bucket and collection").set_defaults(func=cmd_init)
    sub.add_parser("worker", help="run an ingestion worker").set_defaults(func=cmd_worker)
    api = sub.add_parser("api", help="run the HTTP API")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.set_defaults(func=cmd_api)
    enq = sub.add_parser("enqueue", help="queue existing objects in a bucket for ingestion")
    enq.add_argument("--bucket", default=None)
    enq.add_argument("--prefix", default="")
    enq.set_defaults(func=cmd_enqueue)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
