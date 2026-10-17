"""Storage: the Store interface and its Postgres + pgvector implementation."""

from collections.abc import Sequence
from importlib import resources
from typing import Protocol

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from docs_qa.chunking import Chunk
from docs_qa.models import Hit


class Store(Protocol):
    def init(self) -> None: ...

    def get_fingerprint(self, path: str) -> str | None: ...

    def replace(
        self, path: str, fingerprint: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None: ...

    def search(self, vector: list[float], k: int) -> list[Hit]: ...

    def count(self) -> int: ...


_SEARCH_SQL = """
SELECT c.id, d.path, c.heading, c.content, 1 - (c.embedding <=> %(q)s) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> %(q)s
LIMIT %(k)s
"""


class PgStore:
    """One short-lived connection per call. Simple and safe with FastAPI's threadpool;
    a connection pool (psycopg_pool) is the obvious next step under real load."""

    def __init__(self, url: str) -> None:
        self._url = url

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._url)
        register_vector(conn)
        return conn

    def init(self) -> None:
        """Create the extension, tables and HNSW index (idempotent)."""
        ddl = resources.files("docs_qa").joinpath("schema.sql").read_text(encoding="utf-8")
        with psycopg.connect(self._url, autocommit=True) as conn:
            conn.execute(ddl)

    def get_fingerprint(self, path: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT fingerprint FROM documents WHERE path = %s", (path,))
            found = row.fetchone()
        return found[0] if found else None

    def replace(
        self, path: str, fingerprint: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        """Swap a document's chunks in one transaction, so readers never see half a file."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE path = %s", (path,))  # cascades to chunks
            cur.execute(
                "INSERT INTO documents (path, fingerprint) VALUES (%s, %s) RETURNING id",
                (path, fingerprint),
            )
            doc_id = cur.fetchone()[0]
            cur.executemany(
                "INSERT INTO chunks (document_id, ordinal, heading, content, embedding)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (doc_id, c.ordinal, c.heading, c.text, Vector(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def search(self, vector: list[float], k: int) -> list[Hit]:
        with self._connect() as conn:
            rows = conn.execute(_SEARCH_SQL, {"q": Vector(vector), "k": k}).fetchall()
        return [
            Hit(chunk_id=r[0], file=r[1], heading=r[2], text=r[3], score=float(r[4])) for r in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
