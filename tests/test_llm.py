import json

import httpx
import pytest

from docs_qa.llm import (
    NOT_FOUND,
    ExtractiveProvider,
    OpenAIProvider,
    ProviderError,
    build_prompt,
    rank_sentences,
    split_sentences,
    terms,
)
from docs_qa.models import Hit


def hit(text: str, score: float = 0.8, file: str = "a.md", heading: str = "A") -> Hit:
    return Hit(chunk_id=1, file=file, heading=heading, text=text, score=score)


CACHE = hit(
    "uv caches downloaded packages so repeated installs are fast. "
    "Run `uv cache clean` to remove all cache entries. "
    "The cache directory can be changed with UV_CACHE_DIR.",
    file="caching.md",
)
OTHER = hit("Install uv with the standalone installer script on every platform.", 0.5, "install.md")


def test_terms_drop_stopwords_and_stem_inflections():
    assert terms("How do I clear the caches?") == {"clear", "cach"}
    assert terms("clearing cached entries") == {"clear", "cach", "entry"}
    assert terms("removed removes remove") == {"remov"}


def test_split_sentences_skips_code_tables_and_html():
    md = (
        "Real prose sentence number one is here.\n\n"
        "```bash\nuv run this should not appear in any sentence\n```\n\n"
        "| a | b | c | d | e | f |\n|---|---|---|---|---|---|\n\n"
        "<div>html tag</div> Another [linked](http://x.y) prose sentence follows now."
    )
    assert split_sentences(md) == [
        "Real prose sentence number one is here.",
        "html tag Another linked prose sentence follows now.",
    ]


def test_split_sentences_strips_list_markers_and_emphasis_but_keeps_identifiers():
    md = (
        "- `uv cache clean` removes _all_ entries; set UV_CACHE_DIR to move it.\n"
        "- Second item has enough words."
    )
    assert split_sentences(md) == [
        "uv cache clean removes all entries; set UV_CACHE_DIR to move it.",
        "Second item has enough words.",
    ]


def test_rank_sentences_picks_matching_sentence():
    picked = rank_sentences("How do I remove cache entries?", [CACHE, OTHER])
    assert picked[0] == (0, "Run uv cache clean to remove all cache entries.")


def test_rank_sentences_needs_two_matching_terms():
    # only "installer" overlaps: one shared term is not enough for a multi-word question
    assert rank_sentences("Which installer handles proxies behind firewalls?", [OTHER]) == []


def test_rank_sentences_orders_by_hit_then_position():
    picked = rank_sentences("uv cache directory installs", [CACHE], n=3)
    plain = CACHE.text.replace("`", "")
    positions = [plain.index(s) for _, s in picked]
    assert len(picked) >= 2 and positions == sorted(positions)


def test_extractive_answer_has_inline_citation_numbers():
    gen = ExtractiveProvider().generate("How do I remove cache entries?", [OTHER, CACHE])
    text = "".join(gen.stream)
    assert text.endswith("[1]")
    assert [h.file for h in gen.citations] == ["caching.md"]  # OTHER was not used


def test_extractive_streams_sentence_by_sentence():
    gen = ExtractiveProvider().generate("uv cache directory installs", [CACHE])
    parts = list(gen.stream)
    assert len(parts) >= 2 and parts[0][0] != " " and parts[1][0] == " "


def test_extractive_not_found():
    gen = ExtractiveProvider().generate("quantum entanglement of bananas", [CACHE])
    assert "".join(gen.stream) == NOT_FOUND
    assert gen.citations == []


def test_build_prompt_numbers_sources():
    prompt = build_prompt("why?", [CACHE, OTHER])
    assert "[1] caching.md > A" in prompt and "[2] install.md > A" in prompt
    assert prompt.endswith("Question: why?")


def _sse(*tokens: str) -> bytes:
    events = [{"choices": [{"delta": {"content": t}}]} for t in tokens]
    events.insert(1, {"choices": []})  # some servers send empty-choices chunks
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


def test_openai_provider_streams_tokens_and_sends_expected_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse("Hello", " world"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider("sk-test", "https://example.test/v1", "m", client)
    gen = provider.generate("q?", [CACHE])
    assert list(gen.stream) == ["Hello", " world"]
    assert gen.citations == [CACHE]
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["stream"] is True and seen["body"]["model"] == "m"


def test_openai_provider_wraps_http_errors():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    gen = OpenAIProvider("k", "https://example.test/v1", "m", client).generate("q?", [CACHE])
    with pytest.raises(ProviderError):
        list(gen.stream)


def test_openai_provider_requires_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(None, "https://example.test/v1", "m")
