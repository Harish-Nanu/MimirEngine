"""Kafka producer and topic setup shared by the API, CLI and workers."""

from __future__ import annotations

import logging

from .config import Settings
from .errors import TransientError
from .messages import IngestJob

log = logging.getLogger(__name__)


def producer_config(settings: Settings) -> dict:
    return {
        "bootstrap.servers": settings.kafka_bootstrap,
        "acks": "all",
        "enable.idempotence": True,
        "linger.ms": 20,
    }


class JobPublisher:
    """Thread-safe: each batch tracks its own delivery failures."""

    def __init__(self, settings: Settings, producer=None):
        if producer is None:
            from confluent_kafka import Producer

            producer = Producer(producer_config(settings))
        self.producer = producer
        self.settings = settings

    def send(self, topic: str, value: bytes, key: str | None, failures: list[str],
             headers: dict[str, str] | None = None) -> None:
        """Queue one message. Delivery errors are appended to `failures`; call flush() to confirm."""

        def on_delivery(err, msg):
            if err is not None:
                log.error("delivery to %s failed: %s", msg.topic(), err)
                failures.append(str(err))

        self.producer.produce(
            topic, value=value, key=key, headers=list((headers or {}).items()), on_delivery=on_delivery
        )
        self.producer.poll(0)

    def flush(self, failures: list[str], timeout: float = 30.0) -> None:
        """Block until queued messages are acknowledged; raise TransientError if any were not."""
        remaining = self.producer.flush(timeout)
        if remaining:
            raise TransientError(f"{remaining} Kafka message(s) not delivered within {timeout}s")
        if failures:
            raise TransientError(f"Kafka delivery failed: {failures[0]}")

    def publish(self, jobs: list[IngestJob]) -> None:
        failures: list[str] = []
        for job in jobs:
            self.send(self.settings.ingest_topic, job.to_bytes(), job.doc_id, failures)
        self.flush(failures)


def ensure_topics(settings: Settings) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
    existing = admin.list_topics(timeout=30).topics
    wanted = [settings.ingest_topic, settings.retry_topic, settings.dlq_topic]
    new = [
        NewTopic(t, num_partitions=settings.topic_partitions, replication_factor=settings.topic_replication)
        for t in wanted
        if t not in existing
    ]
    if not new:
        return
    for topic, future in admin.create_topics(new).items():
        try:
            future.result()
            log.info("created topic %s", topic)
        except Exception as exc:
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise
