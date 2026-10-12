"""End-to-end test against a running stack (`docker compose up -d`).

Skipped unless MIMIR_STACK_URL is set, e.g.:
    MIMIR_STACK_URL=http://localhost:8000 pytest tests/test_stack.py -v
"""

import os
import time
import uuid

import httpx
import pytest

URL = os.environ.get("MIMIR_STACK_URL")
pytestmark = pytest.mark.skipif(not URL, reason="MIMIR_STACK_URL not set")
TERMINAL = {"ready", "failed"}


@pytest.fixture(scope="module")
def client():
    headers = {"x-api-key": os.environ["MIMIR_API_KEY"]} if os.environ.get("MIMIR_API_KEY") else {}
    with httpx.Client(base_url=URL, headers=headers, timeout=120) as c:
        deadline = time.time() + 120
        while c.get("/ready").status_code != 200:
            assert time.time() < deadline, f"stack not ready: {c.get('/ready').text}"
            time.sleep(2)
        yield c


def wait_for(client, doc_ids, timeout=180):
    deadline = time.time() + timeout
    while True:
        docs = {d: client.get(f"/documents/{d}").json() for d in doc_ids}
        if all(doc["status"] in TERMINAL for doc in docs.values()):
            return docs
        assert time.time() < deadline, f"ingestion did not finish: {docs}"
        time.sleep(2)


def test_ingest_query_ask_delete(client):
    tag = uuid.uuid4().hex[:8]  # unique per run so earlier runs' documents don't interfere
    files = [
        ("files", (f"lighthouse-{tag}.md",
                   f"# Lighthouse {tag}\n\nThe Skerry lighthouse keeper refills the lamp oil every Tuesday."
                   .encode(), "text/markdown")),
        ("files", (f"bakery-{tag}.txt",
                   b"The bakery on Mill Street sells rye sourdough and closes at 3pm on Sundays.", "text/plain")),
        ("files", (f"broken-{tag}.pdf", b"%PDF-1.4 this is not really a pdf", "application/pdf")),
    ]
    r = client.post("/documents", files=files)
    assert r.status_code == 202, r.text
    good, bakery, broken = (d["doc_id"] for d in r.json())

    docs = wait_for(client, [good, bakery, broken])
    assert docs[good]["status"] == "ready" and docs[good]["chunks"] >= 1
    assert docs[bakery]["status"] == "ready"
    assert docs[broken]["status"] == "failed" and "PermanentError" in docs[broken]["error"]

    hits = client.post("/query", json={"query": "when is the lamp oil refilled?", "top_k": 1}).json()["results"]
    assert hits[0]["doc_id"] == good

    scoped = client.post("/query", json={"query": "lamp oil", "doc_id": bakery}).json()["results"]
    assert {h["doc_id"] for h in scoped} == {bakery}

    if client.get("/ready").json()["checks"]["llm"] == "ok":
        r = client.post("/ask", json={"question": f"How often does the Lighthouse {tag} keeper refill the oil?"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert "tuesday" in out["answer"].lower()
        assert any(c["doc_id"] == good for c in out["citations"])

    assert client.delete(f"/documents/{good}").status_code == 204
    assert client.get(f"/documents/{good}").status_code == 404
    hits = client.post("/query", json={"query": "lamp oil", "doc_id": good}).json()["results"]
    assert hits == []
