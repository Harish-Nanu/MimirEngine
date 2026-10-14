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
   ├── query ──▶ API ──▶ embed query ──▶ Qdrant top-k chunks
   └── ask ────▶ API ──▶ top-k chunks ──▶ Claude ──▶ answer + citations
```

Each worker: download from MinIO → extract text → chunk → embed in batches → replace the
document's chunks in Qdrant.

| Component | Role |
|---|---|
| `mimir/api.py` | FastAPI: upload, document status/list/delete, `/query`, `/ask`, `/stats`, health, web UI |
| `mimir/answer.py` | Answers questions from retrieved chunks with Claude, with chunk-level citations |
| `mimir/status.py` | Per-document ingestion status (queued → processing → ready / retrying / failed) |
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

Open http://localhost:8000 for the web UI: upload files, watch their status, and ask questions.
Answers need an LLM: set `ANTHROPIC_API_KEY` in `.env` for Claude, or run a free local model with
Ollama (see [Answer providers](#answer-providers)). Without either, everything else still works.

Or use the API. Ingest files:

```bash
curl -F files=@report.pdf -F files=@notes.md http://localhost:8000/documents
```

Check progress (`queued`, `processing`, `ready`, `retrying` or `failed`, with the error):

```bash
curl -s localhost:8000/documents/<doc_id>
curl -s 'localhost:8000/documents?status=failed'
curl -s localhost:8000/stats
```

Ask a question. The answer cites the chunks it used, and `sources` lists everything retrieved:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "what did the report conclude?"}'
```

Search without generating an answer:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "what did the report conclude?", "top_k": 5}'
```

Pass `"doc_id"` to `/ask` or `/query` to stay within one document. `DELETE /documents/<doc_id>`
removes a document's chunks, file and status. Interactive API docs: http://localhost:8000/docs.
MinIO console: http://localhost:9001 (`minioadmin` / `minioadmin`).

`GET /health` is a liveness check. `GET /ready` checks MinIO, Kafka and Qdrant and returns 503 if
any is down.

### Security

Set `MIMIR_API_KEY` to require `X-API-Key: <key>` (or `Authorization: Bearer <key>`) on every
endpoint except `/health`, `/ready` and the UI page. Paste the key into the UI's key field.
Uploads are capped at `MIMIR_MAX_UPLOAD_MB` (default 100), and unsupported file types are refused
with 415 before anything is stored. Change the MinIO credentials before exposing the stack.

### Bulk ingestion

For large batches, copy files straight into the bucket (e.g. `mc cp --recursive`), then queue them:

```bash
docker compose run --rm api enqueue --prefix archive/2024/
```

Doc IDs derive from bucket and key, both here and for API uploads, so running `enqueue` again
re-ingests the same files in place without creating duplicates. If an upload returned 503 because
Kafka was down, the files are already stored: `enqueue --prefix uploads/` queues them.

### Scaling

```bash
docker compose up -d --scale worker=10
```

Workers share the `mimir-workers` consumer group. Parallelism is capped by the number of topic
partitions (`MIMIR_TOPIC_PARTITIONS`, default 12). Consumer lag on `mimir.ingest` is the metric to
autoscale on (e.g. KEDA's Kafka scaler on Kubernetes).

### Inspecting and replaying the DLQ

Failed documents show up in `GET /documents?status=failed` with the error. To read the raw topic:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 --topic mimir.ingest.dlq --from-beginning
```

After fixing the cause (an outage, a quota, a bad key), put dead-lettered jobs back on the queue:

```bash
docker compose run --rm api replay-dlq                   # everything
docker compose run --rm api replay-dlq --skip-permanent  # not corrupt/unsupported files
```

## Configuration

All settings are environment variables with the `MIMIR_` prefix. See `mimir/config.py` for the full
list and `.env.example` for the common ones.

| Provider | `MIMIR_EMBEDDING_PROVIDER` | Default model | Needs key |
|---|---|---|---|
| sentence-transformers (in-process) | `local` | `BAAI/bge-small-en-v1.5` (384d) | no |
| OpenAI | `openai` | `text-embedding-3-small` | yes |
| Voyage AI | `voyage` | `voyage-3.5` | yes |

### Answer providers

`MIMIR_LLM_PROVIDER` picks what writes answers for `/ask`:

| Provider | `MIMIR_LLM_PROVIDER` | Needs | Citations |
|---|---|---|---|
| Claude | `claude` (default) | `ANTHROPIC_API_KEY` | Native, per chunk |
| Ollama (local) | `ollama` | Ollama running on the host | Model cites `[n]` markers, mapped to chunks |

**Claude** uses `claude-opus-5-5` at `medium` effort (`MIMIR_LLM_MODEL`, `MIMIR_LLM_EFFORT`).
Requests opt into server-side fallbacks, so a request Claude's safety classifiers decline is retried
on a fallback model instead of failing.

**Ollama** runs on the host rather than in Compose, so it can use the host GPU; containers reach it
at `host.docker.internal:11434`. Install it from https://ollama.com, then:

```bash
ollama pull qwen3:4b-instruct   # default MIMIR_OLLAMA_MODEL; fits in 4 GB of VRAM
```

Set `MIMIR_LLM_PROVIDER=ollama` in `.env` and restart the API. `GET /ready` reports whether Ollama is
reachable and the model is pulled. Small local models answer less reliably than Claude; a larger
model (`MIMIR_OLLAMA_MODEL`) helps if you have the memory for it.

Switching provider or model changes the vector size. Use a new `MIMIR_QDRANT_COLLECTION` when you do.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests use in-memory fakes for MinIO and Kafka, qdrant-client's in-process mode for Qdrant, and
a mocked HTTP transport for Claude. No services are needed. To test a running stack end to end:

```bash
MIMIR_STACK_URL=http://localhost:8000 pytest tests/test_stack.py -v
```

 Running the CLI on the host (`mimir worker`, `mimir api`) against the
compose services works with the default settings.

## Limitations

- Each document is read fully into worker memory; very large files need workers sized for them.
- Scanned PDFs without a text layer are dead-lettered (no OCR).
- `GET /documents` lists in ID order, not by upload time.
