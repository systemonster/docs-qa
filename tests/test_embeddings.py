import json
import math

import httpx
import pytest

from docs_qa.embeddings import EMBED_DIM, HashEmbedder, OpenAIEmbedder


def cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_is_deterministic_unit_length_and_right_size():
    e = HashEmbedder()
    v1, v2 = e.embed(["clear the cache"])[0], e.embed_query("clear the cache")
    assert v1 == v2 and len(v1) == EMBED_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0)


def test_hash_embedder_prefers_lexically_similar_text():
    e = HashEmbedder()
    q = e.embed_query("clear the cache")
    close, far = e.embed_query("how to clear cache entries"), e.embed_query("install python")
    assert cos(q, close) > cos(q, far)


def test_hash_embedder_handles_empty_text():
    assert not any(HashEmbedder().embed_query(""))


def test_openai_embedder_batches_and_restores_order():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        assert body["dimensions"] == EMBED_DIM
        rows = [
            {"index": i, "embedding": [float(i + 1)] + [0.0] * (EMBED_DIM - 1)}
            for i, _ in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": rows[::-1]})  # reversed: embedder must sort

    client = httpx.Client(transport=httpx.MockTransport(handler))
    e = OpenAIEmbedder("k", "https://example.test/v1/", "m", client)
    out = e.embed([str(i) for i in range(70)])
    assert calls == [64, 6]
    assert len(out) == 70 and all(math.isclose(v[0], 1.0) for v in out)  # all normalised


def test_openai_embedder_requires_key():
    with pytest.raises(ValueError):
        OpenAIEmbedder("", "https://example.test/v1", "m")
