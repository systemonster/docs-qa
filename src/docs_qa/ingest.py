"""Read Markdown files, chunk, embed and store them. Re-running is idempotent.

python -m docs_qa.ingest [PATH ...]      # files or folders; default: DOCS_PATHS
"""

import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from docs_qa.chunking import chunk_markdown
from docs_qa.config import Settings, get_settings
from docs_qa.db import PgStore, Store
from docs_qa.embeddings import EMBED_DIM, Embedder, build_embedder


@dataclass
class IngestReport:
    files: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    chunks: int = 0


def find_markdown(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.rglob("*.md")))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"not found: {raw}")
    return found


def display_path(p: Path) -> str:
    """Stable id for a file: relative to the working directory when possible."""
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def fingerprint(content: bytes, embedder: Embedder, max_words: int, overlap: int) -> str:
    """Hash of everything that affects the stored vectors, not only the file content.

    Changing the embedder or chunk size therefore re-ingests unchanged files.
    """
    h = hashlib.sha256(content)
    h.update(f"|{embedder.name}|{max_words}|{overlap}".encode())
    return h.hexdigest()


def ingest(
    paths: list[str], store: Store, embedder: Embedder, max_words: int = 400, overlap: int = 60
) -> IngestReport:
    report = IngestReport()
    for file in find_markdown(paths):
        report.files += 1
        raw = file.read_bytes()
        path, fp = display_path(file), fingerprint(raw, embedder, max_words, overlap)
        previous = store.get_fingerprint(path)
        if previous == fp:
            report.unchanged += 1
            continue
        chunks = chunk_markdown(raw.decode("utf-8", errors="replace"), max_words, overlap)
        vectors = embedder.embed([c.embed_text for c in chunks]) if chunks else []
        if vectors and len(vectors[0]) != EMBED_DIM:
            raise ValueError(
                f"embedder returned {len(vectors[0])} dims, schema expects {EMBED_DIM}"
            )
        store.replace(path, fp, chunks, vectors)
        report.chunks += len(chunks)
        if previous is None:
            report.added += 1
        else:
            report.updated += 1
    return report


def run(paths: list[str] | None = None, settings: Settings | None = None) -> IngestReport:
    s = settings or get_settings()
    store = PgStore(s.database_url)
    store.init()
    return ingest(
        paths or s.doc_paths,
        store,
        build_embedder(s),
        s.chunk_max_words,
        s.chunk_overlap_words,
    )


if __name__ == "__main__":
    print(asdict(run(sys.argv[1:] or None)))
