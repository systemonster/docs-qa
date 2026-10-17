import math
import os
from collections.abc import Sequence

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from docs_qa.chunking import Chunk
from docs_qa.config import Settings
from docs_qa.db import PgStore
from docs_qa.embeddings import HashEmbedder
from docs_qa.models import Hit

DOCS = {
    "caching.md": """# Caching

uv caches downloaded packages so that repeated installs are fast.

## Clearing the cache

Run `uv cache clean` to remove all cache entries. Use `uv cache prune` to remove only
entries that are no longer needed. The cache directory can be changed with UV_CACHE_DIR.
""",
    "install.md": """# Installation

Install uv with the standalone installer script. On macOS and Linux use curl to download
the installer. On Windows use the PowerShell installer. You can also install from PyPI.

## Upgrading

Run `uv self update` to upgrade uv to the latest version.
""",
    "python.md": """# Python versions

uv can install and manage several Python interpreters side by side. Pin a project to a
version by writing it into the .python-version file at the project root.
""",
}


class MemoryStore:
    """In-memory Store with real cosine ranking, so API tests need no Postgres."""

    def __init__(self) -> None:
        self.docs: dict[str, tuple[str, list[Chunk], list[list[float]]]] = {}

    def init(self) -> None:
        pass

    def get_fingerprint(self, path: str) -> str | None:
        return self.docs[path][0] if path in self.docs else None

    def replace(
        self, path: str, fingerprint: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        self.docs[path] = (fingerprint, list(chunks), list(vectors))

    def search(self, vector: list[float], k: int) -> list[Hit]:
        rows = [
            (sum(a * b for a, b in zip(vector, v, strict=True)), path, c)
            for path, (_, chunks, vecs) in self.docs.items()
            for c, v in zip(chunks, vecs, strict=True)
        ]
        rows.sort(key=lambda r: -r[0])
        return [
            Hit(chunk_id=i, file=p, heading=c.heading, text=c.text, score=s)
            for i, (s, p, c) in enumerate(rows[:k])
            if not math.isnan(s)
        ]

    def count(self) -> int:
        return sum(len(d[1]) for d in self.docs.values())


@pytest.fixture
def docs_dir(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, body in DOCS.items():
        (folder / name).write_text(body, encoding="utf-8")
    return folder


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def memory_store():
    return MemoryStore()


@pytest.fixture
def settings(docs_dir):
    return Settings(_env_file=None, docs_paths=str(docs_dir), embedder="hash")


TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://docs_qa:docs_qa@localhost:5432/docs_qa_test"
)


@pytest.fixture(scope="session")
def pg_url():
    """Create the throwaway test database, or skip DB tests if Postgres is not reachable."""
    info = conninfo_to_dict(TEST_DB_URL)
    admin = make_conninfo(TEST_DB_URL, dbname="postgres")
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (info["dbname"],))
            if not exists.fetchone():
                conn.execute(f'CREATE DATABASE "{info["dbname"]}"')
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable ({exc.__class__.__name__}); start it: make db")
    return TEST_DB_URL


@pytest.fixture
def pg_store(pg_url):
    store = PgStore(pg_url)
    store.init()
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
    return store


@pytest.fixture(params=["memory", "postgres"])
def any_store(request):
    """Run the same test against the in-memory store and real Postgres + pgvector."""
    return request.getfixturevalue("memory_store" if request.param == "memory" else "pg_store")
