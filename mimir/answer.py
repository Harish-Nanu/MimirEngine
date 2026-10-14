"""Answer questions from retrieved chunks, citing the chunks used.

Two providers share one interface, chosen by MIMIR_LLM_PROVIDER:

- claude: each hit is passed as a `search_result` content block with citations enabled,
  so citations come back pointing at specific chunks.
- ollama: a local model. It has no native citations, so hits are numbered in the prompt,
  the model is told to cite them as [n], and those markers are mapped back to chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from .config import Settings
from .errors import FatalError, RateLimitError, TransientError
from .vectorstore import SearchHit

SYSTEM_PROMPT = """You answer questions using only the search results provided with each question. \
They are passages retrieved from the user's document collection.

- Base every claim on the search results and cite them. Do not use outside knowledge to fill gaps.
- If the results don't contain the answer, say so plainly and, if useful, say what they do cover.
- If results disagree, point out the disagreement instead of picking one silently.
- Be concise. Lead with the answer; skip preamble."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Citation:
    doc_id: str
    chunk_index: int
    filename: str
    cited_text: str
    ref: int | None = None  # the [n] marker in the answer text, for providers that use them


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    stop_reason: str | None = None


class ClaudeAnswerer:
    def __init__(self, settings: Settings, client=None):
        if client is None:
            import anthropic

            try:
                client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key \
                    else anthropic.Anthropic()
            except anthropic.AnthropicError as exc:
                raise FatalError(f"Claude is not configured (set ANTHROPIC_API_KEY): {exc}") from exc
            if not (client.api_key or client.auth_token or getattr(client, "credentials", None)):
                raise FatalError("Claude is not configured: set ANTHROPIC_API_KEY")
        self.client = client
        self.settings = settings

    def ping(self) -> None:
        """Credentials are checked at construction; no request is spent on a health check."""

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        import anthropic

        if not hits:
            return Answer(text="No indexed documents matched this question.")

        content = [
            {
                "type": "search_result",
                "source": f"{h.doc_id}#{h.chunk_index}",
                "title": h.metadata.get("filename") or h.doc_id,
                "content": [{"type": "text", "text": h.text}],
                "citations": {"enabled": True},
            }
            for h in hits
        ]
        content.append({"type": "text", "text": question})

        try:
            response = self.client.beta.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": self.settings.llm_effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise RateLimitError("Claude rate limit", _retry_after(exc.response)) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise FatalError(f"Claude rejected the API key: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(f"Claude API error {exc.status_code}") from exc
            raise FatalError(f"Claude request rejected ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise TransientError(f"could not reach Claude: {exc}") from exc

        if response.stop_reason == "refusal":
            return Answer(text="Claude declined to answer this question.", model=response.model,
                          stop_reason="refusal")

        parts: list[str] = []
        citations: list[Citation] = []
        seen: set[tuple[int, int, int]] = set()
        for block in response.content:
            if block.type != "text":
                continue
            parts.append(block.text)
            for c in block.citations or []:
                if c.type != "search_result_location" or not 0 <= c.search_result_index < len(hits):
                    continue
                k = (c.search_result_index, c.start_block_index, c.end_block_index)
                if k in seen:
                    continue
                seen.add(k)
                h = hits[c.search_result_index]
                citations.append(Citation(h.doc_id, h.chunk_index, h.metadata.get("filename", ""), c.cited_text))

        return Answer(text="".join(parts).strip(), citations=citations, model=response.model,
                      stop_reason=response.stop_reason)


def _retry_after(resp) -> float | None:
    try:
        return float(resp.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None


OLLAMA_SYSTEM_PROMPT = SYSTEM_PROMPT + """
- Cite sources inline with their bracketed number, like [1] or [2][3], right after the claim they support.
- Only cite source numbers that appear in the question."""

CITE_MARK = re.compile(r"\[(\d+)\]")
THINK_BLOCK = re.compile(r"^.*?</think>", re.DOTALL)


class OllamaAnswerer:
    """A model served by a local Ollama instance (https://ollama.com)."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.model = settings.ollama_model
        self.client = client or httpx.Client(base_url=settings.ollama_url, timeout=300.0)

    def ping(self) -> None:
        try:
            tags = self.client.get("/api/tags").json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TransientError(f"could not reach Ollama at {self.settings.ollama_url}: {exc}") from exc
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise FatalError(f"Ollama model {self.model!r} is not pulled; run `ollama pull {self.model}`")

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(text="No indexed documents matched this question.")

        sources = "\n\n".join(
            f"[{i}] {h.metadata.get('filename') or h.doc_id}\n{h.text}" for i, h in enumerate(hits, 1)
        )
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": self.settings.ollama_num_ctx},
            "messages": [
                {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Search results:\n\n{sources}\n\nQuestion: {question}"},
            ],
        }
        try:
            resp = self.client.post("/api/chat", json=payload)
        except httpx.TransportError as exc:
            raise TransientError(f"could not reach Ollama at {self.settings.ollama_url}: {exc}") from exc
        if resp.status_code == 404:
            raise FatalError(f"Ollama model {self.model!r} is not pulled; run `ollama pull {self.model}`")
        if resp.status_code >= 500:
            raise TransientError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise FatalError(f"Ollama rejected the request ({resp.status_code}): {resp.text[:300]}")

        body = resp.json()
        # Thinking-only models ignore "think": false and put their reasoning before </think>.
        text = THINK_BLOCK.sub("", body["message"]["content"]).strip()
        citations, seen = [], set()
        for m in CITE_MARK.finditer(text):
            n = int(m.group(1))
            if n in seen or not 1 <= n <= len(hits):
                continue
            seen.add(n)
            h = hits[n - 1]
            citations.append(Citation(h.doc_id, h.chunk_index, h.metadata.get("filename", ""), h.text, ref=n))
        citations.sort(key=lambda c: c.ref)
        return Answer(text=text, citations=citations, model=f"ollama/{self.model}",
                      stop_reason=body.get("done_reason"))


def build_answerer(settings: Settings):
    """The configured answer provider. Raises FatalError when it can't be set up."""
    provider = settings.llm_provider.lower()
    if provider == "claude":
        return ClaudeAnswerer(settings)
    if provider == "ollama":
        return OllamaAnswerer(settings)
    raise FatalError(f"unknown MIMIR_LLM_PROVIDER {settings.llm_provider!r} (use claude or ollama)")
