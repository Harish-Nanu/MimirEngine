import uuid

from mimir.status import DELETED, QUEUED, READY, DocumentStatus

from .fakes import memory_status
from .test_pipeline import SETTINGS, make_job


def test_put_keeps_created_at_and_list_hides_deleted():
    store = memory_status(SETTINGS)
    job = make_job()
    store.put(DocumentStatus.for_job(job, QUEUED))
    created = store.get(job.doc_id).created_at

    store.put(DocumentStatus.for_job(job, READY, chunks=4))
    rec = store.get(job.doc_id)
    assert rec.status == READY and rec.chunks == 4 and rec.created_at == created

    other = make_job(doc_id="6f1c2a4e-9d3b-4c8a-b0e5-2f7d9a1c3e58")
    store.put(DocumentStatus.for_job(other, DELETED))
    docs, cursor = store.list()
    assert [d.doc_id for d in docs] == [job.doc_id] and cursor is None
    assert [d.doc_id for d in store.list(DELETED)[0]] == [other.doc_id]
    assert store.counts()[READY] == 1 and store.counts()[DELETED] == 1


def test_list_paginates():
    store = memory_status(SETTINGS)
    for _ in range(5):
        store.put(DocumentStatus.for_job(make_job(doc_id=str(uuid.uuid4())), QUEUED))
    page, cursor = store.list(limit=2)
    seen = {d.doc_id for d in page}
    while cursor:
        page, cursor = store.list(limit=2, cursor=cursor)
        seen |= {d.doc_id for d in page}
    assert len(seen) == 5
