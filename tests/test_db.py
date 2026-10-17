"""Postgres-specific checks. Skipped automatically when the test database is unreachable."""

import psycopg

from docs_qa.chunking import Chunk
from docs_qa.embeddings import EMBED_DIM


def test_schema_is_idempotent_and_has_hnsw_index(pg_store, pg_url):
    pg_store.init()  # second run must not fail
    with psycopg.connect(pg_url) as conn:
        indexdef = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'"
        ).fetchone()[0]
    assert "USING hnsw" in indexdef and "vector_cosine_ops" in indexdef


def test_hnsw_index_is_used_for_nearest_neighbour_query(pg_store, pg_url):
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    pg_store.replace("a.md", "fp", [Chunk("A", "text " * 5, 0)], [vec])
    with psycopg.connect(pg_url) as conn:
        conn.execute("SET enable_seqscan = off")  # a 1-row table would otherwise seq-scan
        plan = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 1",
                (str(vec),),
            )
        )
    assert "chunks_embedding_hnsw" in plan


def test_deleting_a_document_cascades_to_chunks(pg_store, pg_url):
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    pg_store.replace("a.md", "fp", [Chunk("A", "text " * 5, 0)], [vec])
    assert pg_store.count() == 1
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("DELETE FROM documents WHERE path = 'a.md'")
    assert pg_store.count() == 0


def test_cosine_score_of_identical_vector_is_one(pg_store):
    vec = [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)
    pg_store.replace("a.md", "fp", [Chunk("A", "text " * 5, 0)], [vec])
    (hit,) = pg_store.search(vec, 3)
    assert abs(hit.score - 1.0) < 1e-6 and hit.file == "a.md" and hit.heading == "A"
