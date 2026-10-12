"""Answer questions from retrieved chunks with Claude, citing the chunks it used.

Each hit is passed as a `search_result` content block with citations enabled, so the
answer comes back with citations that point at specific chunks rather than free text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
