"""Command line entry point: `mimir init | worker | api | enqueue | replay-dlq`."""

from __future__ import annotations

import argparse
import logging
import os

from dataclasses import replace

from .config import get_settings
from .messages import IngestJob, doc_id_for_object

log = logging.getLogger("mimir.cli")


def cmd_init(args) -> None:
    """Create Kafka topics, the MinIO bucket and the Qdrant collection (idempotent)."""
    from .embeddings import build_embedder
    from .queue import ensure_topics
    from .status import QdrantStatusStore
    from .storage import MinioStore
    from .vectorstore import QdrantStore

    settings = get_settings()
    ensure_topics(settings)
    MinioStore(settings).ensure_bucket(settings.s3_bucket)
    QdrantStore(settings).ensure_collection(build_embedder(settings).dimension)
    QdrantStatusStore(settings).ensure_collection()
    logging.info("init complete")


def cmd_worker(args) -> None:
    from .embeddings import build_embedder
    from .pipeline import Pipeline
    from .queue import JobPublisher
    from .status import QdrantStatusStore
    from .storage import MinioStore
    from .vectorstore import QdrantStore
    from .worker import Worker

    settings = get_settings()
    embedder = build_embedder(settings)
    vectors = QdrantStore(settings)
    vectors.ensure_collection(embedder.dimension)
    status = QdrantStatusStore(settings)
    status.ensure_collection()
    pipeline = Pipeline(settings, MinioStore(settings), embedder, vectors)
    Worker(settings, pipeline, JobPublisher(settings), status=status).run()


def cmd_api(args) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


def cmd_enqueue(args) -> None:
    """Queue every object under a bucket prefix. Doc ids derive from bucket/key, so re-running is safe."""
    from .queue import JobPublisher
    from .status import QUEUED, DocumentStatus, QdrantStatusStore
    from .storage import MinioStore

    settings = get_settings()
    bucket = args.bucket or settings.s3_bucket
    publisher = JobPublisher(settings)
    status = QdrantStatusStore(settings)
    batch: list[IngestJob] = []
    total = 0

    def flush():
        nonlocal batch, total
        status.put(*(DocumentStatus.for_job(j, QUEUED) for j in batch))
        publisher.publish(batch)
        total += len(batch)
        batch = []

    for key in MinioStore(settings).list_keys(bucket, args.prefix):
        batch.append(IngestJob(doc_id=doc_id_for_object(bucket, key), bucket=bucket, key=key,
                               filename=os.path.basename(key)))
        if len(batch) >= 500:
            flush()
    if batch:
        flush()
    log.info("queued %d object(s) from s3://%s/%s", total, bucket, args.prefix)


def cmd_replay_dlq(args) -> None:
    """Move dead-lettered jobs back onto the ingest topic with a fresh attempt count.

    Uses its own consumer group, so each DLQ message is replayed once. Stops when the DLQ is drained.
    """
    from confluent_kafka import Consumer

    from .errors import PermanentError
    from .queue import JobPublisher
    from .status import QUEUED, DocumentStatus, QdrantStatusStore

    settings = get_settings()
    publisher = JobPublisher(settings)
    status = QdrantStatusStore(settings)
    consumer = Consumer({
        "bootstrap.servers": settings.kafka_bootstrap,
        "group.id": "mimir-dlq-replay",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([settings.dlq_topic])
    replayed = skipped = idle = 0
    try:
        while idle < args.idle_timeout and (not args.limit or replayed < args.limit):
            msg = consumer.poll(1.0)
            if msg is None:
                idle += 1
                continue
            idle = 0
            if msg.error():
                log.warning("consumer error: %s", msg.error())
                continue
            try:
                job = IngestJob.from_bytes(msg.value())
            except PermanentError as exc:
                log.warning("skipping malformed DLQ message at offset %d: %s", msg.offset(), exc)
                skipped += 1
                consumer.commit(message=msg, asynchronous=False)
                continue
            if args.skip_permanent and (job.last_error or "").startswith("PermanentError"):
                skipped += 1
                consumer.commit(message=msg, asynchronous=False)
                continue
            fresh = replace(job, attempt=0, not_before=0.0)
            status.put(DocumentStatus.for_job(fresh, QUEUED))
            publisher.publish([fresh])
            consumer.commit(message=msg, asynchronous=False)
            replayed += 1
    finally:
        consumer.close()
    log.info("replayed %d job(s), skipped %d", replayed, skipped)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=os.environ.get("MIMIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
    rep = sub.add_parser("replay-dlq", help="re-queue dead-lettered jobs")
    rep.add_argument("--limit", type=int, default=0, help="stop after N jobs (0 = all)")
    rep.add_argument("--skip-permanent", action="store_true",
                     help="drop jobs that failed with PermanentError (corrupt/unsupported files)")
    rep.add_argument("--idle-timeout", type=int, default=10, help="seconds without messages before stopping")
    rep.set_defaults(func=cmd_replay_dlq)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
