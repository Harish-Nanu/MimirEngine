"""Kafka consumer that runs the pipeline and routes failures.

Delivery is at-least-once: an offset is committed only after the job is stored in Qdrant
or safely handed to the retry/DLQ topic. Qdrant writes are idempotent, so redelivery is harmless.

Routing:
  success                     -> commit
  PermanentError              -> DLQ
  other error, attempts left  -> retry topic, with exponential backoff (not_before)
  other error, out of attempts-> DLQ
  FatalError                  -> stop the worker without committing

Each step is mirrored to the status store. Jobs for documents deleted through the API are skipped.
"""

from __future__ import annotations

import logging
import random
import signal
import time
from dataclasses import dataclass, replace

from .config import Settings
from .errors import FatalError, PermanentError
from .messages import IngestJob
from .pipeline import Pipeline
from .queue import JobPublisher
from .status import DELETED, FAILED, PROCESSING, READY, RETRYING, DocumentStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    topic: str
    job: IngestJob


def route_failure(job: IngestJob, exc: Exception, settings: Settings) -> Route:
    """Pick where a failed job goes. FatalError is not handled here; it propagates."""
    error = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, PermanentError):
        return Route(settings.dlq_topic, replace(job, last_error=error))
    if job.attempt + 1 >= settings.max_attempts:
        return Route(settings.dlq_topic, replace(job, attempt=job.attempt + 1, last_error=error))
    delay = min(settings.retry_max_delay_s, settings.retry_base_delay_s * 2**job.attempt)
    delay *= 0.75 + random.random() / 2  # jitter so retried bursts spread out
    return Route(settings.retry_topic, job.next_attempt(error, delay))


class Worker:
    def __init__(self, settings: Settings, pipeline: Pipeline, publisher: JobPublisher, consumer=None,
                 status=None):
        self.settings = settings
        self.pipeline = pipeline
        self.publisher = publisher
        self.consumer = consumer
        self.status = status  # QdrantStatusStore-like, optional
        self._paused: dict[tuple[str, int], float] = {}
        self._running = True

    def stop(self, *_):
        log.info("shutdown requested")
        self._running = False

    def run(self) -> None:
        from confluent_kafka import Consumer

        if self.consumer is None:
            self.consumer = Consumer(
                {
                    "bootstrap.servers": self.settings.kafka_bootstrap,
                    "group.id": self.settings.consumer_group,
                    "enable.auto.commit": False,
                    "auto.offset.reset": "earliest",
                    # One large PDF can take a while; don't get kicked from the group mid-job.
                    "max.poll.interval.ms": 900_000,
                }
            )
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.consumer.subscribe(
            [self.settings.ingest_topic, self.settings.retry_topic],
            on_assign=lambda c, parts: self._paused.clear(),
            on_revoke=lambda c, parts: self._paused.clear(),
        )
        log.info("worker started")
        try:
            while self._running:
                self._resume_due()
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    log.warning("consumer error: %s", msg.error())
                    continue
                self.handle(msg)
        finally:
            self.consumer.close()

    def handle(self, msg) -> None:
        try:
            job = IngestJob.from_bytes(msg.value())
        except PermanentError as exc:
            self._dispatch(msg, lambda f: self.publisher.send(
                self.settings.dlq_topic, msg.value(), None, f, headers={"mimir-error": str(exc)}))
            return

        if job.not_before > time.time():
            self._defer(msg, job.not_before)
            return

        if self._is_deleted(job.doc_id):
            log.info("doc=%s was deleted, skipping job", job.doc_id)
            self._commit(msg)
            return
        self._mark(job, PROCESSING)

        try:
            chunks = self.pipeline.process(job)
            if self.status is not None:
                if self._is_deleted(job.doc_id):  # deleted while we were processing it
                    self.pipeline.vectors.delete_document(job.doc_id)
                else:
                    # Inside the try: if this fails the job is retried, which is idempotent.
                    self.status.put(DocumentStatus.for_job(job, READY, chunks=chunks))
        except FatalError:
            raise
        except Exception as exc:
            route = route_failure(job, exc, self.settings)
            retrying = route.topic == self.settings.retry_topic
            log.log(logging.WARNING if retrying else logging.ERROR,
                    "doc=%s attempt=%d -> %s: %s", job.doc_id, job.attempt, route.topic, exc)
            if self._dispatch(msg, lambda f: self.publisher.send(
                    route.topic, route.job.to_bytes(), route.job.doc_id, f)):
                self._mark(route.job, RETRYING if retrying else FAILED, error=route.job.last_error)
            return

        self._commit(msg)

    def _is_deleted(self, doc_id: str) -> bool:
        if self.status is None:
            return False
        try:
            current = self.status.get(doc_id)
        except Exception as exc:  # status is advisory here; don't block ingestion on it
            log.warning("status lookup failed for doc=%s: %s", doc_id, exc)
            return False
        return current is not None and current.status == DELETED

    def _mark(self, job: IngestJob, status: str, **kw) -> None:
        """Best-effort status update for intermediate states."""
        if self.status is None:
            return
        try:
            self.status.put(DocumentStatus.for_job(job, status, **kw))
        except Exception as exc:
            log.warning("status update to %s failed for doc=%s: %s", status, job.doc_id, exc)

    def _dispatch(self, msg, send) -> bool:
        """Send to retry/DLQ, and commit only once Kafka has acknowledged it."""
        failures: list[str] = []
        try:
            send(failures)
            self.publisher.flush(failures)
        except Exception as exc:
            log.error("could not route failed job, will reprocess: %s", exc)
            self._rewind(msg)
            time.sleep(5)
            return False
        self._commit(msg)
        return True

    def _commit(self, msg) -> None:
        self.consumer.commit(message=msg, asynchronous=False)

    def _defer(self, msg, until: float) -> None:
        """Retry message not due yet: rewind to it and pause its partition until it is."""
        from confluent_kafka import TopicPartition

        self._rewind(msg)
        self.consumer.pause([TopicPartition(msg.topic(), msg.partition())])
        self._paused[(msg.topic(), msg.partition())] = until

    def _rewind(self, msg) -> None:
        from confluent_kafka import TopicPartition

        self.consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))

    def _resume_due(self) -> None:
        if not self._paused:
            return
        from confluent_kafka import TopicPartition

        now = time.time()
        due = [tp for tp, until in self._paused.items() if until <= now]
        if due:
            self.consumer.resume([TopicPartition(t, p) for t, p in due])
            for tp in due:
                del self._paused[tp]
