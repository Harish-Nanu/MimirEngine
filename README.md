# MimirEngine
MimirEngine: Mass-Scale RAG Ingestion Workflow
MimirEngine is an enterprise-grade document ingestion and retrieval system. It bridges scalable data engineering with modern AI by orchestrating an asynchronous workflow that transforms thousands of raw files into semantic vector representations.

Unlike basic RAG scripts that process documents synchronously, MimirEngine is built to handle heavy, unpredictable workloads. By leveraging Kafka as a message broker and MinIO for S3-compatible object storage, the system gracefully handles API rate limits, implements dead-letter queues for corrupted files, and dynamically scales worker nodes to ensure no data is lost during ingestion spikes.

## Architecture

```
            upload                       ingest job
 client ───────────▶ API ──▶ MinIO       ┌────────────────────────────┐
   │                  │                  │                            ▼
   │                  └──▶ Kafka: mimir.ingest ──▶ workers (N) ──▶ Qdrant
   │                         mimir.ingest.retry ◀──┤   │
   │                         mimir.ingest.dlq   ◀──────┘
   └── query ──▶ API ──▶ embed query ──▶ Qdrant top-k chunks
```

Each worker: download from MinIO → extract text → chunk → embed in batches → replace the
document's chunks in Qdrant.

| Component | Role |
|---|---|
| `mimir/api.py` | FastAPI: `POST /documents`, `POST /query`, `GET /health` |
| `mimir/worker.py` | Kafka consumer, failure routing, delayed retries |
| `mimir/pipeline.py` | Per-document processing and embedding backoff |
| `mimir/embeddings.py` | Pluggable providers: `local`, `openai`, `voyage` |
| `mimir/extract.py`, `mimir/chunking.py` | Text extraction (txt/md/csv/json/html/pdf) and chunking |
| `mimir/storage.py`, `mimir/vectorstore.py` | MinIO and Qdrant adapters |

## Reliability

- **At-least-once delivery.** A Kafka offset is committed only after the chunks are written to
  Qdrant, or after the job has been acknowledged by the retry/DLQ topic. Workers can crash at any
  point without losing a document.
- **Idempotent writes.** Chunk IDs derive from `doc_id` and chunk index, and re-ingesting a document
  replaces its old chunks, so redelivered jobs never create duplicates.
- **Rate limits.** Embedding calls are retried in-process with backoff, honouring `Retry-After`.
  `MIMIR_EMBEDDING_MAX_RPS` adds a client-side cap per worker.
- **Delayed retries.** When in-process retries run out, the job goes to `mimir.ingest.retry` with
  exponential backoff. Workers pause that partition until the job is due, rather than sleeping.
- **Dead-letter queue.** Corrupted, encrypted, empty or unsupported files go straight to
  `mimir.ingest.dlq`. So do jobs that exhaust `MIMIR_MAX_ATTEMPTS`. The `last_error` field says why.
- **Misconfiguration stops workers.** A rejected API key or a vector-size mismatch stops the
  worker without committing. A bad key won't send your whole corpus to the DLQ.

## Quick start

Requires Docker with Compose.

```bash
cp .env.example .env            # optional; defaults use the local embedding model
docker compose up -d --build    # kafka, minio, qdrant, init, api, 3 workers
```

Ingest files:

```bash
curl -F files=@report.pdf -F files=@notes.md http://localhost:8000/documents
```

Query:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "what did the report conclude?", "top_k": 5}'
```

Pass `"doc_id"` to search within one document. Interactive API docs: http://localhost:8000/docs.
MinIO console: http://localhost:9001 (`minioadmin` / `minioadmin`).

### Bulk ingestion

For large batches, copy files straight into the bucket (e.g. `mc cp --recursive`), then queue them:

```bash
docker compose run --rm api enqueue --prefix archive/2024/
```

Doc IDs derive from bucket and key, so running `enqueue` again re-ingests the same files in place
without creating duplicates.

### Scaling

```bash
docker compose up -d --scale worker=10
```

Workers share the `mimir-workers` consumer group. Parallelism is capped by the number of topic
partitions (`MIMIR_TOPIC_PARTITIONS`, default 12). Consumer lag on `mimir.ingest` is the metric to
autoscale on (e.g. KEDA's Kafka scaler on Kubernetes).

### Inspecting the DLQ

```bash
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 --topic mimir.ingest.dlq --from-beginning
```

## Configuration

All settings are environment variables with the `MIMIR_` prefix. See `mimir/config.py` for the full
list and `.env.example` for the common ones.

| Provider | `MIMIR_EMBEDDING_PROVIDER` | Default model | Needs key |
|---|---|---|---|
| sentence-transformers (in-process) | `local` | `BAAI/bge-small-en-v1.5` (384d) | no |
| OpenAI | `openai` | `text-embedding-3-small` | yes |
| Voyage AI | `voyage` | `voyage-3.5` | yes |

Switching provider or model changes the vector size. Use a new `MIMIR_QDRANT_COLLECTION` when you do.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests use in-memory fakes for MinIO and Kafka, and qdrant-client's in-process mode for Qdrant.
No services are needed. Running the CLI on the host (`mimir worker`, `mimir api`) against the
compose services works with the default settings.

## Limitations

- Each document is read fully into worker memory; very large files need workers sized for them.
- Scanned PDFs without a text layer are dead-lettered (no OCR).
- No ingestion-status endpoint yet; follow progress through worker logs and the DLQ.
