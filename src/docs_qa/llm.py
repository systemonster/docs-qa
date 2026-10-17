"""Answer providers. Each turns (question, retrieved hits) into a token stream plus citations.

- extractive: no LLM. Picks the sentences that best match the question. Default, deterministic.
- openai: any OpenAI-compatible /chat/completions endpoint, streamed over SSE with httpx.
- anthropic: Anthropic Messages API through the official SDK (optional extra).
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from docs_qa.config import Settings
from docs_qa.models import Hit

NOT_FOUND = "I could not find an answer in the indexed documents."


class ProviderError(RuntimeError):
    """The upstream LLM failed; the API maps this to 502 / a WebSocket error message."""


@dataclass
class Generation:
    stream: Iterator[str]
    citations: list[Hit]  # citations[i] is referred to as [i + 1] in the answer


class Provider(Protocol):
    def generate(self, question: str, hits: list[Hit]) -> Generation: ...


# ---------------------------------------------------------------- extractive

_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
        "use",
        "used",
        "using",
        "not",
        "this",
        "that",
        "there",
        "should",
        "would",
    }
)


def _stem(word: str) -> str:
    """Very rough stemmer, enough to match clear/clearing and cache/caches/cached."""
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + replacement
            break
    return word.removesuffix("e")


def terms(text: str) -> set[str]:
    """Stemmed content words of a text (stop words removed)."""
    return {_stem(w) for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOP}


def split_sentences(markdown: str) -> list[str]:
    """Prose sentences of a Markdown chunk. Code blocks, tables and HTML are dropped."""
    text = re.sub(r"(```|~~~).*?\1", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"<[^>]+>|[`*]|^\s*(!!!|///).*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)  # _emphasis_, not UV_CACHE_DIR
    blocks = re.split(r"\n\s*\n|\n(?=\s*(?:[-+]|\d+\.)\s)", text)
    out: list[str] = []
    for block in blocks:
        if block.lstrip().startswith("|"):
            continue
        flat = re.sub(r"^([-+]|\d+\.)\s+", "", " ".join(block.split()))  # list marker
        for sentence in re.split(r"(?<=[.!?])\s+", flat):
            if 5 <= len(sentence.split()) <= 60:
                out.append(sentence)
    return out


def rank_sentences(question: str, hits: list[Hit], n: int = 3) -> list[tuple[int, str]]:
    """Best `n` (hit index, sentence) pairs, ordered by hit then position in the chunk.

    Score = share of question terms the sentence contains, plus a small bonus for sentences
    from better-ranked chunks. A sentence needs at least two matching terms (or one, if the
    question has only one content word), so incidental overlap does not produce an answer.
    """
    q = terms(question)
    if not q:
        return []
    need = min(2, len(q))
    scored: list[tuple[float, int, int, str]] = []
    seen: set[str] = set()
    for hit_idx, hit in enumerate(hits):
        for pos, sentence in enumerate(split_sentences(hit.text)):
            matched = len(q & terms(sentence))
            if matched < need or sentence in seen:
                continue
            seen.add(sentence)
            scored.append((matched / len(q) + 0.1 * hit.score, hit_idx, pos, sentence))
    best = sorted(scored, key=lambda t: (-t[0], t[1], t[2]))[:n]
    return [(h, s) for _, h, _, s in sorted(best, key=lambda t: (t[1], t[2]))]


class ExtractiveProvider:
    def generate(self, question: str, hits: list[Hit]) -> Generation:
        picked = rank_sentences(question, hits)
        if not picked:
            return Generation(iter([NOT_FOUND]), [])
        cited = sorted({h for h, _ in picked})
        number = {hit_idx: i + 1 for i, hit_idx in enumerate(cited)}
        sentences = [f"{s} [{number[h]}]" for h, s in picked]
        stream = (s if i == 0 else " " + s for i, s in enumerate(sentences))
        return Generation(stream, [hits[i] for i in cited])


# ---------------------------------------------------------------- LLM providers

_SYSTEM = (
    "You answer questions about documentation. Use only the numbered sources provided. "
    "Cite sources inline as [1], [2]. If the sources do not contain the answer, say so."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{i}] {h.file} > {h.heading}\n{h.text}" for i, h in enumerate(hits, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


class OpenAIProvider:
    def __init__(
        self, api_key: str | None, base_url: str, model: str, client: httpx.Client | None = None
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=120)

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        return Generation(self._stream(question, hits), hits)

    def _stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        body = {
            "model": self._model,
            "stream": True,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
        }
        try:
            with self._client.stream("POST", self._url, headers=self._headers, json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    choices = json.loads(data).get("choices") or [{}]
                    if token := choices[0].get("delta", {}).get("content"):
                        yield token
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc


class AnthropicProvider:
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        try:
            import anthropic
        except ImportError as exc:
            raise ValueError("install the extra: pip install 'docs-qa[anthropic]'") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        return Generation(self._stream(question, hits), hits)

    def _stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{"role": "user", "content": build_prompt(question, hits)}],
            ) as stream:
                yield from stream.text_stream
        except self._anthropic.APIError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc


def build_provider(s: Settings) -> Provider:
    if s.llm_provider == "openai":
        return OpenAIProvider(s.openai_api_key, s.openai_base_url, s.openai_chat_model)
    if s.llm_provider == "anthropic":
        return AnthropicProvider(s.anthropic_api_key, s.anthropic_model)
    return ExtractiveProvider()
