# docs-qa

A small FastAPI service that answers questions over a folder of Markdown files and tells you
which file and section each answer came from.

It is a learning/portfolio project: about 900 lines of source (plus 600 of tests), two Postgres
tables, no auth, no frontend. It has not been used in production.

- **Ingest** Markdown, split by headings into overlapping chunks, embed, store in **Postgres + pgvector** (HNSW index). Re-ingesting unchanged files does nothing.
- **Search** `GET /search?q=...` returns the closest chunks with score, file and heading.
- **Ask** `POST /ask` returns an answer plus numbered citations. `WS /ws/ask` streams the same answer token by token.
- **Pluggable parts:** embeddings (`fastembed` local ONNX by default, `openai`-compatible, or an offline `hash` baseline) and answer providers (`extractive` with no LLM by default, `anthropic`, `openai`).
- **Eval:** `python -m docs_qa.eval` prints retrieval hit@k and MRR for a question set.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
      MD[Markdown files] --> CH[chunking.py<br/>split by headings, window + overlap]
      CH --> EM1[Embedder]
      EM1 --> DB[(Postgres + pgvector<br/>documents, chunks, HNSW)]
    end
    subgraph Query
      Q[question] --> EM2[Embedder<br/>query mode]
      EM2 --> DB
      DB -->|top-k chunks| P{Provider}
      P -->|extractive| S[best sentences + citations]
      P -->|anthropic / openai| L[LLM stream + citations]
    end
    API[FastAPI<br/>/ingest /search /ask /ws/ask] --- Ingest
    API --- Query
```

| File | Role |
|---|---|
| `src/docs_qa/chunking.py` | Markdown to chunks (pure functions) |
| `src/docs_qa/embeddings.py` | `Embedder` protocol, fastembed / openai / hash implementations |
| `src/docs_qa/db.py` + `schema.sql` | `Store` protocol, Postgres implementation, DDL with HNSW index |
| `src/docs_qa/ingest.py` | Walk files, fingerprint, chunk, embed, store (CLI + function) |
| `src/docs_qa/search.py` | Embed the query, ask the store for the top k |
| `src/docs_qa/llm.py` | Providers: extractive, OpenAI-compatible (SSE over httpx), Anthropic |
| `src/docs_qa/api.py` | FastAPI app factory, REST and WebSocket endpoints |
| `src/docs_qa/eval.py` | hit@k and MRR |
| `evals/questions.jsonl` | 15 questions with the file that should be found |
| `sample_docs/` | Copy of the uv docs (MIT / Apache-2.0), see `sample_docs/SOURCE.md` |

## Quickstart

```bash
docker compose up -d --build        # Postgres (pgvector) + the API on :8000
curl -X POST localhost:8000/ingest  # indexes sample_docs/ and this README
```

```bash
curl 'localhost:8000/search?q=how+do+I+clear+the+cache&k=3'

curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How do I clear the cache?"}'

# streaming
echo '{"question": "How do I clear the cache?"}' | websocat -n1 ws://localhost:8000/ws/ask
```

Without Docker for the API (Postgres still needs the pgvector extension):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"             # add ".[dev,anthropic]" for the Anthropic provider
cp .env.example .env
make db                             # docker compose up -d --wait db
make ingest && make run
```

Configuration is environment variables only, see `.env.example`.

### WebSocket protocol

`WS /ws/ask` takes JSON text frames and keeps the connection open for more questions.

```text
client -> server  {"question": "How do I clear the cache?", "k": 5}          k is optional (1-20)
server -> client  {"type": "token", "text": "Run uv cache clean ... [1]"}    zero or more
                  {"type": "citations", "citations": [
                      {"n": 1, "file": "sample_docs/concepts/cache.md",
                       "heading": "Caching > Clearing the cache", "score": 0.81, "snippet": "..."}]}
                  {"type": "done"}
bad request       {"type": "error", "message": "invalid request: ..."}      connection stays open
```

With the extractive provider a `token` message is one sentence; with an LLM provider it is
whatever the model streams. `[n]` in the text refers to `citations[n - 1]`.

## Design decisions and trade-offs

**Why pgvector and not a dedicated vector database.** The data is small and I already want
Postgres for the metadata. Keeping chunks, documents and vectors in one transactional store
means re-ingesting a file is one transaction (old chunks out, new ones in) and I do not run a
second service. The cost: pgvector's HNSW is good, but a dedicated engine gives more tuning,
filtering and scale options. For tens of millions of vectors I would revisit this.

**HNSW, not ivfflat.** Both are approximate nearest-neighbour indexes. ivfflat clusters the
vectors into lists and needs data already in the table to build good clusters. HNSW builds a
layered graph, needs no training step, and has better recall at the same speed, at the price of
slower builds and more memory. For a table that fills up incrementally HNSW is the simpler
choice. The query uses `<=>` (cosine distance) with `vector_cosine_ops`; an operator/opclass
mismatch silently falls back to a sequential scan, and `tests/test_db.py` checks the plan.

**Chunking.** Split at headings (levels 1-3) so a chunk is one topic, keep the heading path as
a breadcrumb (`Caching > Clearing the cache`) and prepend it to the text that gets embedded.
Sections longer than 400 words are cut on line boundaries into windows with 60 words of
overlap, so a sentence at a boundary is complete in at least one chunk. Words stand in for
tokens (about 1.3 tokens per English word, so 400 words is roughly 500-550 tokens; code-heavy
text has more tokens per word). The numbers are settings, not the result of a tuning study.
Cutting on lines rather than sentences keeps code blocks and lists intact.

**Idempotent ingest.** Each file gets a fingerprint: sha256 of the content plus the embedder
name and chunk settings. Same fingerprint, skip. A changed file is replaced in one
transaction. Including the embedder in the fingerprint matters: swapping the model must
re-embed everything, otherwise old and new vectors would be mixed in one index.

**Extractive fallback.** The default answer provider does no generation: it scores the
sentences of the retrieved chunks by how many question terms they contain and returns the best
three, each tagged with a citation number. Reasons: it needs no API key, it is deterministic
(so tests and the demo are stable), it cannot hallucinate (every word is from the docs), and it
makes retrieval quality easy to see. The cost is obvious: it quotes rather than explains, and
it cannot combine facts across sentences. The LLM providers are there for that.

**WebSocket vs SSE.** Both stream server to client. SSE is plain HTTP, simpler, and
reconnects by itself, but it is one-way and one request at a time. A WebSocket is a
bidirectional connection that stays open, so one connection can carry many questions and the
protocol can grow (cancel, feedback) without new endpoints. For one-way token streaming SSE
would honestly be enough; I used WebSocket because the project is meant to exercise it.

**Fixed 384 dimensions.** The column is `vector(384)`. The default model
(`BAAI/bge-small-en-v1.5`) produces 384; the OpenAI embedder asks `text-embedding-3-*` for 384
via the `dimensions` parameter. A model with another size needs a schema change and a full
re-ingest; ingest fails loudly instead of storing the wrong size.

## Evaluation

`python -m docs_qa.eval --ingest` embeds the corpus (`sample_docs/` and this README), runs the
15 questions in `evals/questions.jsonl`, and checks at which rank the expected file appears.
Chunks are collapsed to files first (first occurrence wins). hit@k is the share of questions whose
file is within the top k files. MRR is the mean of 1/rank (0 if not found within the top 10 chunks).

Results from a real run (Postgres 16.2 + pgvector, 208 chunks, 2026-10-17):

| Embedder | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| `fastembed` `BAAI/bge-small-en-v1.5` (default) | 0.87 | 1.00 | 1.00 | 0.92 |
| `hash` (lexical baseline, no model) | 0.67 | 0.87 | 0.87 | 0.74 |

Reproduce with `make eval` (default embedder) or `EMBEDDER=hash make eval`. The `hash` row shows
what the model buys: it misses questions where the question and the docs use different words
(for example "different platforms" versus "universal resolution"). `fastembed` puts the
right file first for 13 of 15 questions; the other two land at rank 3 ("add an extra package
index") and rank 2 (the README question about re-embedding).

15 questions is a smoke test, not a benchmark: one question moves a metric by about 7 points,
and I wrote the questions after reading the docs.

## Tests

```bash
make lint && make test
```

61 tests (88% line coverage): chunking, ranking, extractive answers, OpenAI/embedding clients against mocked HTTP,
the REST API and WebSocket through `TestClient`, and Postgres tests (schema, HNSW plan, ingest
idempotency, search). The Postgres tests use `TEST_DATABASE_URL` (default
`postgresql://docs_qa:docs_qa@localhost:5432/docs_qa_test`, created automatically) and are
skipped with a message if the database is not reachable (53 pass and 8 skip without Postgres). The tests use the offline hash
embedder, so no model is downloaded in CI.

## Limitations

- No authentication, rate limiting or request size limits. `POST /ingest` cannot take a path
  from the request (it reads `DOCS_PATHS`), but anyone who can reach the API can trigger it.
- A new database connection per call. Fine for a demo; use a pool (`psycopg_pool`) under load.
- Deleted files are not removed from the index (no prune step).
- Retrieval is vector-only. No keyword/hybrid search and no reranker, so exact identifiers
  (a flag name, an error code) can lose to semantically similar text.
- The default embedding model is English-oriented.
- The extractive provider quotes sentences; it does not synthesize. Its sentence splitter is
  regex-based and drops tables and code blocks, so answers that live in code blocks are missed.
- The `anthropic` provider has not been run against the real API (no key was used while
  building this). The `openai` provider and embedder are tested against a mocked HTTP server
  only. Treat all three as unverified against the live services.
- Eval is small, and the corpus is third-party documentation, not a realistic private knowledge base.
