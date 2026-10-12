"""Runs the real QdrantStore against qdrant-client's in-process mode (no server needed)."""

import uuid

import pytest
from qdrant_client import QdrantClient

from mimir.config import Settings
from mimir.embeddings import HashEmbedder
from mimir.errors import FatalError
from mimir.vectorstore import Chunk, QdrantStore


@pytest.fixture
def store():
    s = QdrantStore(Settings(_env_file=None))
    s.client = QdrantClient(location=":memory:")
    s.ensure_collection(256)
    return s


def chunks(doc_id, texts):
    emb = HashEmbedder()
    return [Chunk(doc_id, i, t, emb.embed_query(t), {"filename": "f.txt"}) for i, t in enumerate(texts)]


def test_replace_search_and_filter(store):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    store.replace_document(a, chunks(a, ["kafka brokers", "minio buckets", "qdrant vectors"]))
    store.replace_document(b, chunks(b, ["bananas are yellow"]))

    (hit,) = store.search(HashEmbedder().embed_query("yellow bananas"), top_k=1)
    assert hit.doc_id == b and hit.text == "bananas are yellow" and hit.metadata == {"filename": "f.txt"}

    hits = store.search(HashEmbedder().embed_query("yellow bananas"), top_k=10, doc_id=a)
    assert {h.doc_id for h in hits} == {a} and len(hits) == 3

    store.replace_document(a, chunks(a, ["only one now"]))
    assert len(store.search(HashEmbedder().embed_query("x"), top_k=10, doc_id=a)) == 1


def test_ensure_collection_is_idempotent_and_checks_dimension(store):
    store.ensure_collection(256)
    with pytest.raises(FatalError, match="dimension"):
        store.ensure_collection(384)


def test_delete_document(store):
    a = str(uuid.uuid4())
    store.replace_document(a, chunks(a, ["one", "two"]))
    store.delete_document(a)
    assert store.search(HashEmbedder().embed_query("one"), top_k=10, doc_id=a) == []
