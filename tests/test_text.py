import io

import pytest
from pypdf import PdfWriter

from mimir.chunking import chunk_text
from mimir.errors import PermanentError
from mimir.extract import extract_text
from mimir.messages import IngestJob, doc_id_for_object


def test_chunks_cover_text_and_respect_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(40))
    chunks = chunk_text(text, size=500, overlap=80)
    assert all(len(c) <= 500 for c in chunks)
    assert len(chunks) > 1
    for i in range(40):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_chunks_overlap_and_terminate_on_unbroken_text():
    chunks = chunk_text("x" * 2500, size=1000, overlap=200)
    assert [len(c) for c in chunks] == [1000, 1000, 900]


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_chunk_args_validated(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size=size, overlap=overlap)


def test_extract_plain_and_markdown():
    assert extract_text(b"# Title\nbody", "a.md") == "# Title\nbody"
    assert extract_text("caf\xe9".encode("latin-1"), "a.txt") == "caf\xe9"


def test_extract_html_drops_scripts():
    html = b"<html><script>evil()</script><h1>Title</h1><p>Body   text</p></html>"
    assert extract_text(html, "page.html") == "Title\nBody text"


def test_extract_json_pretty_prints():
    assert '"a": 1' in extract_text(b'{"a":1}', "x.json")


@pytest.mark.parametrize(
    "data,name",
    [
        (b"%PDF-1.4 garbage", "broken.pdf"),
        (b"\x00\x01\x02binary", "fake.txt"),
        (b"   \n ", "empty.txt"),
        (b"MZ\x90\x00", "program.exe"),
        (b"{not json", "x.json"),
    ],
)
def test_bad_files_are_permanent_errors(data, name):
    with pytest.raises(PermanentError):
        extract_text(data, name)


def test_pdf_without_text_is_permanent():
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.write(buf)
    with pytest.raises(PermanentError, match="no extractable text"):
        extract_text(buf.getvalue(), "blank.pdf")


def test_job_roundtrip_and_malformed():
    job = IngestJob(doc_id="d", bucket="b", key="k", filename="f.txt")
    assert IngestJob.from_bytes(job.to_bytes()) == job
    with pytest.raises(PermanentError):
        IngestJob.from_bytes(b"not json")
    with pytest.raises(PermanentError):
        IngestJob.from_bytes(b'{"unexpected": 1}')


def test_doc_id_is_stable_per_object():
    assert doc_id_for_object("b", "k") == doc_id_for_object("b", "k")
    assert doc_id_for_object("b", "k") != doc_id_for_object("b", "k2")
