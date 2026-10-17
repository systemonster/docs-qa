import pytest
from fastapi.testclient import TestClient

from docs_qa.api import create_app
from docs_qa.llm import ExtractiveProvider, Generation, ProviderError


@pytest.fixture
def client(settings, memory_store, embedder):
    app = create_app(settings, memory_store, embedder, ExtractiveProvider())
    with TestClient(app) as c:  # the context manager runs the lifespan (store.init)
        assert c.post("/ingest").json()["added"] == 3
        yield c


def test_health_reports_chunk_count(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] >= 3


def test_ingest_is_idempotent_over_http(client):
    again = client.post("/ingest").json()
    assert (again["added"], again["unchanged"]) == (0, 3)


def test_ingest_missing_path_is_a_400(settings, memory_store, embedder):
    bad = settings.model_copy(update={"docs_paths": "does/not/exist"})
    with TestClient(create_app(bad, memory_store, embedder, ExtractiveProvider())) as c:
        assert c.post("/ingest").status_code == 400


def test_search_returns_ranked_chunks(client):
    hits = client.get("/search", params={"q": "clear the cache", "k": 2}).json()
    assert len(hits) == 2
    assert hits[0]["file"].endswith("caching.md") and "Clearing" in hits[0]["heading"]
    assert hits[0]["score"] >= hits[1]["score"]


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 99}, {}])
def test_search_validates_input(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_ask_returns_answer_with_citations(client):
    r = client.post("/ask", json={"question": "How do I remove cache entries?"})
    body = r.json()
    assert r.status_code == 200
    assert "uv cache clean" in body["answer"] and "[1]" in body["answer"]
    assert body["citations"][0]["file"].endswith("caching.md")
    assert body["citations"][0]["n"] == 1


def test_ask_without_support_says_so_and_cites_nothing(client):
    body = client.post("/ask", json={"question": "quantum entanglement of bananas"}).json()
    assert body["citations"] == [] and "could not find" in body["answer"]


def test_ask_validates_body(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_maps_provider_failure_to_502(settings, memory_store, embedder):
    class Failing:
        def generate(self, question, hits):
            def boom():
                raise ProviderError("upstream down")
                yield ""

            return Generation(boom(), [])

    with TestClient(create_app(settings, memory_store, embedder, Failing())) as c:
        c.post("/ingest")
        r = c.post("/ask", json={"question": "anything"})
    assert r.status_code == 502 and "upstream down" in r.json()["detail"]


def test_websocket_streams_tokens_then_citations_then_done(client):
    with client.websocket_connect("/ws/ask") as ws:
        ws.send_json({"question": "How do I remove cache entries?"})
        messages = []
        while True:
            messages.append(ws.receive_json())
            if messages[-1]["type"] == "done":
                break
    types = [m["type"] for m in messages]
    assert types[-2:] == ["citations", "done"] and set(types[:-2]) == {"token"}
    streamed = "".join(m["text"] for m in messages if m["type"] == "token")
    assert "uv cache clean" in streamed
    assert messages[-2]["citations"][0]["file"].endswith("caching.md")


def test_websocket_survives_bad_request_and_serves_next_question(client):
    with client.websocket_connect("/ws/ask") as ws:
        ws.send_json({"question": ""})
        assert ws.receive_json()["type"] == "error"
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"question": "How do I upgrade uv?"})
        assert ws.receive_json()["type"] == "token"
