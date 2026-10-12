"""Split text into overlapping chunks, preferring paragraph, line, then sentence boundaries."""

from __future__ import annotations

SEPARATORS = ("\n\n", "\n", ". ", " ")


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            end = _best_break(text, start, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # Step back by `overlap`, but always make forward progress.
        start = max(end - overlap, start + 1)
    return chunks


def _best_break(text: str, start: int, end: int) -> int:
    """Latest separator in the back half of the window, so chunks don't get tiny."""
    floor = start + (end - start) // 2
    for sep in SEPARATORS:
        idx = text.rfind(sep, floor, end)
        if idx != -1:
            return idx + len(sep)
    return end
