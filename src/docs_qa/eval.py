"""Retrieval evaluation: does the expected source file show up in the top results?

    python -m docs_qa.eval [--questions evals/questions.jsonl] [--ingest]

Each line of the questions file is {"question": "...", "source": "path/of/expected/file.md"}.
Chunks are collapsed to files (first occurrence wins) before ranking, because the question is
"did we find the right document", not "the right paragraph".
"""

import argparse
import json
from pathlib import Path

from docs_qa import ingest as ingest_mod
from docs_qa.config import get_settings
from docs_qa.db import PgStore, Store
from docs_qa.embeddings import Embedder, build_embedder
from docs_qa.search import retrieve

DEPTH = 10  # chunks fetched per question
CUTOFFS = (1, 3, 5)


def file_rank(files_in_order: list[str], expected: str) -> int | None:
    """1-based rank of `expected` among distinct files, or None if absent."""
    distinct = list(dict.fromkeys(files_in_order))
    return distinct.index(expected) + 1 if expected in distinct else None


def summarize(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in CUTOFFS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def evaluate(questions: list[dict], store: Store, embedder: Embedder) -> list[int | None]:
    ranks = []
    for q in questions:
        hits = retrieve(store, embedder, q["question"], DEPTH)
        ranks.append(file_rank([h.file for h in hits], q["source"]))
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--ingest", action="store_true", help="ingest DOCS_PATHS first")
    args = parser.parse_args()

    s = get_settings()
    store, embedder = PgStore(s.database_url), build_embedder(s)
    store.init()
    if args.ingest:
        print("ingest:", ingest_mod.run(settings=s))
    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    ranks = evaluate(questions, store, embedder)
    for q, r in zip(questions, ranks, strict=True):
        print(f"{'miss' if r is None else f'#{r}':>5}  {q['question']}")
    print(f"\nembedder={embedder.name} questions={len(questions)} chunks={store.count()}")
    print(" ".join(f"{k}={v:.2f}" for k, v in summarize(ranks).items()))


if __name__ == "__main__":
    main()
