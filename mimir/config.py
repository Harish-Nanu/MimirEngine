"""Runtime configuration, read from environment variables prefixed with MIMIR_."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MIMIR_", env_file=".env", extra="ignore")

    # Kafka
    kafka_bootstrap: str = "localhost:29092"  # host listener from docker-compose
    ingest_topic: str = "mimir.ingest"
    retry_topic: str = "mimir.ingest.retry"
    dlq_topic: str = "mimir.ingest.dlq"
    consumer_group: str = "mimir-workers"
    topic_partitions: int = 12  # upper bound on useful worker replicas
    topic_replication: int = 1

    # Retry policy for transient failures (rate limits, network errors)
    max_attempts: int = 5
    retry_base_delay_s: float = 5.0
    retry_max_delay_s: float = 300.0

    # MinIO / S3
    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_secure: bool = False
    s3_bucket: str = "mimir-raw"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "mimir_chunks"

    # Embeddings: "local" | "openai" | "voyage" | "hash" (hash is for tests only)
    embedding_provider: str = "local"
    embedding_model: str | None = None  # provider default when unset
    embedding_api_key: str | None = None
    embedding_batch_size: int = 64
    embedding_max_rps: float = 0.0  # client-side requests/sec cap per worker; 0 = unlimited

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150


@lru_cache
def get_settings() -> Settings:
    return Settings()
