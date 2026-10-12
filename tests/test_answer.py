"""ClaudeAnswerer against the real Anthropic SDK, with HTTP mocked at the transport."""

import json
import uuid

import anthropic
import httpx2
import pytest

from mimir.answer import ClaudeAnswerer
from mimir.errors import FatalError, RateLimitError, TransientError
from mimir.pipeline import Pipeline
from mimir.vectorstore import SearchHit

from .test_pipeline import SETTINGS, api_client, make_job

DOC_A, DOC_B = str(uuid.uuid4()), str(uuid.uuid4())
HITS = [
    SearchHit(0.9, DOC_A, 0, "Kafka retains messages for seven days by default.", {"filename": "kafka.md"}),
    SearchHit(0.7, DOC_B, 3, "MinIO is S3 compatible.", {"filename": "minio.txt"}),
]


def claude_response(**overrides):
    body = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
        "content": [
            {"type": "text", "text": "Kafka keeps messages for seven days.", "citations": [{
                "type": "search_result_location", "cited_text": HITS[0].text, "source": f"{DOC_A}#0",
                "title": "kafka.md", "search_result_index": 0, "start_block_index": 0, "end_block_index": 1,
            }]},
        ],
    }
    body.update(overrides)
    return body


def answerer(handler):
    sent = []

    def record(request):
        sent.append(request)
        return handler(request)

    client = anthropic.Anthropic(api_key="sk-test", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(record)))
    return ClaudeAnswerer(SETTINGS, client=client), sent


def test_answer_sends_search_results_and_maps_citations():
    a, sent = answerer(lambda r: httpx2.Response(200, json=claude_response()))
    ans = a.answer("How long does Kafka keep messages?", HITS)

    assert ans.text == "Kafka keeps messages for seven days."
    (c,) = ans.citations
    assert (c.doc_id, c.chunk_index, c.filename) == (DOC_A, 0, "kafka.md")

    (req,) = sent
    body = json.loads(req.content)
    assert body["model"] == SETTINGS.llm_model and body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req.headers["anthropic-beta"]
    blocks = body["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["search_result", "search_result", "text"]
    assert blocks[1]["title"] == "minio.txt" and blocks[1]["citations"] == {"enabled": True}
    assert blocks[-1]["text"] == "How long does Kafka keep messages?"


def test_no_hits_skips_claude():
    a, sent = answerer(lambda r: httpx2.Response(500))
    assert "No indexed documents" in a.answer("anything", []).text
    assert sent == []


def test_refusal_is_reported_not_raised():
    a, _ = answerer(lambda r: httpx2.Response(200, json=claude_response(stop_reason="refusal", content=[])))
    ans = a.answer("q", HITS)
    assert ans.stop_reason == "refusal" and ans.citations == []


@pytest.mark.parametrize("status,headers,exc", [
    (429, {"retry-after": "9"}, RateLimitError),
    (529, {}, TransientError),
    (401, {}, FatalError),
    (400, {}, FatalError),
])
def test_error_mapping(status, headers, exc):
    err = {"type": "error", "error": {"type": "x", "message": "nope"}}
    a, _ = answerer(lambda r: httpx2.Response(status, headers=headers, json=err))
    with pytest.raises(exc) as info:
        a.answer("q", HITS)
    if status == 429:
        assert info.value.retry_after == 9.0


def test_missing_credentials_disable_answerer(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(FatalError, match="not configured"):
        ClaudeAnswerer(SETTINGS.model_copy(update={"anthropic_api_key": None}))


def test_ask_endpoint_end_to_end():
    a, _ = answerer(lambda r: httpx2.Response(200, json=claude_response()))
    with api_client(answerer=a)[0] as client:
        svc = client.app.state.services
        svc.objects.objects[("raw", "k.md")] = HITS[0].text.encode()
        Pipeline(SETTINGS, svc.objects, svc.embedder, svc.vectors).process(
            make_job(doc_id=DOC_A, key="k.md", filename="kafka.md"))

        r = client.post("/ask", json={"question": "How long does Kafka keep messages?"})
        assert r.status_code == 200
        out = r.json()
        assert out["answer"].startswith("Kafka keeps") and out["model"] == "claude-opus-5-5"
        assert out["sources"][0]["doc_id"] == DOC_A


def test_ask_without_claude_is_503():
    with api_client()[0] as client:
        assert client.post("/ask", json={"question": "q"}).status_code == 503
