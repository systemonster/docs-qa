"""HTTP + WebSocket API.

Run:  uvicorn docs_qa.api:create_app --factory

WebSocket protocol (WS /ws/ask), JSON text frames. The connection stays open, so a client
can send several questions one after another.

  client -> server   {"question": "How do I clear the cache?", "k": 5}        (k optional, 1-20)
  server -> client   {"type": "token", "text": "..."}                          (zero or more)
                     {"type": "citations", "citations": [{"n": 1, "file": "...", "heading": "...",
                                                          "score": 0.82, "snippet": "..."}]}
                     {"type": "done"}
  on a bad request   {"type": "error", "message": "..."}   (connection stays open)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from docs_qa.config import Settings, get_settings
from docs_qa.db import PgStore, Store
from docs_qa.embeddings import Embedder, build_embedder
from docs_qa.ingest import IngestReport, ingest
from docs_qa.llm import Provider, ProviderError, build_provider
from docs_qa.models import AskRequest, AskResponse, Hit, to_citations
from docs_qa.search import retrieve


def create_app(
    settings: Settings | None = None,
    store: Store | None = None,
    embedder: Embedder | None = None,
    provider: Provider | None = None,
) -> FastAPI:
    """Dependencies are injectable so tests can run without Postgres or a model download."""
    s = settings or get_settings()
    store = store or PgStore(s.database_url)
    embedder = embedder or build_embedder(s)
    provider = provider or build_provider(s)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.init()
        yield

    app = FastAPI(title="docs-qa", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "chunks": store.count()}

    @app.post("/ingest")
    def post_ingest() -> IngestReport:
        """Ingest the folders configured in DOCS_PATHS. The request cannot choose paths."""
        try:
            return ingest(s.doc_paths, store, embedder, s.chunk_max_words, s.chunk_overlap_words)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/search")
    def search(q: str = Query(min_length=1, max_length=2000), k: int = Query(5, ge=1, le=20)):
        return retrieve(store, embedder, q, k)

    @app.post("/ask")
    def ask(req: AskRequest) -> AskResponse:
        hits = retrieve(store, embedder, req.question, req.k)
        gen = provider.generate(req.question, hits)
        try:
            answer = "".join(gen.stream)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return AskResponse(answer=answer, citations=to_citations(gen.citations))

    @app.websocket("/ws/ask")
    async def ws_ask(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                try:
                    req = AskRequest.model_validate(await ws.receive_json())
                except ValueError as exc:  # bad JSON or failed validation
                    await ws.send_json({"type": "error", "message": f"invalid request: {exc}"})
                    continue
                hits: list[Hit] = await run_in_threadpool(
                    retrieve, store, embedder, req.question, req.k
                )
                gen = provider.generate(req.question, hits)
                try:
                    async for token in iterate_in_threadpool(gen.stream):
                        await ws.send_json({"type": "token", "text": token})
                except ProviderError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                    continue
                citations = [c.model_dump() for c in to_citations(gen.citations)]
                await ws.send_json({"type": "citations", "citations": citations})
                await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            return

    return app
