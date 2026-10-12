"""Turn raw file bytes into plain text. Anything we can't read is a PermanentError."""

from __future__ import annotations

import io
import json
import os
from html.parser import HTMLParser

from .errors import PermanentError

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".log"}
HTML_EXTENSIONS = {".html", ".htm"}


def is_supported(filename: str, content_type: str | None = None) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    ctype = (content_type or "").split(";")[0].strip().lower()
    return (
        ext in TEXT_EXTENSIONS | HTML_EXTENSIONS | {".pdf", ".json"}
        or ctype in ("application/pdf", "application/json")
        or ctype.startswith("text/")
    )


def extract_text(data: bytes, filename: str, content_type: str | None = None) -> str:
    ext = os.path.splitext(filename)[1].lower()
    ctype = (content_type or "").split(";")[0].strip().lower()

    if ext == ".pdf" or ctype == "application/pdf":
        text = _extract_pdf(data)
    elif ext in HTML_EXTENSIONS or ctype == "text/html":
        text = _extract_html(_decode(data))
    elif ext == ".json" or ctype == "application/json":
        text = _extract_json(_decode(data))
    elif ext in TEXT_EXTENSIONS or ctype.startswith("text/"):
        text = _decode(data)
    else:
        raise PermanentError(f"unsupported file type: {filename!r} ({content_type or 'unknown'})")

    text = text.strip()
    if not text:
        raise PermanentError(f"no extractable text in {filename!r}")
    return text


def _decode(data: bytes) -> str:
    if b"\x00" in data[:8192]:
        raise PermanentError("file looks binary or corrupted (NUL bytes in text file)")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PermanentError("PDF is encrypted")
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except PermanentError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise PermanentError(f"corrupted PDF: {exc}") from exc


class _TextOnly(HTMLParser):
    SKIP = {"script", "style", "noscript", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def _extract_html(html: str) -> str:
    parser = _TextOnly()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def _extract_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except ValueError as exc:
        raise PermanentError(f"invalid JSON: {exc}") from exc
