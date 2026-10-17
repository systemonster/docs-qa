-- Idempotent: safe to run on every start.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    fingerprint  TEXT NOT NULL,          -- sha256(content + embedder + chunk params)
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    heading      TEXT NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    UNIQUE (document_id, ordinal)
);

-- HNSW with cosine distance: approximate, but needs no training step and has good recall.
-- The query operator (<=>) must match the opclass (vector_cosine_ops) or the index is unused.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
