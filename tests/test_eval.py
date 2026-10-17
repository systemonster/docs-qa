from docs_qa.eval import evaluate, file_rank, summarize
from docs_qa.ingest import ingest


def test_file_rank_collapses_chunks_to_distinct_files():
    assert file_rank(["a.md", "a.md", "b.md", "c.md"], "b.md") == 2
    assert file_rank(["a.md"], "z.md") is None


def test_summarize_hit_at_k_and_mrr():
    s = summarize([1, 2, None, 5])
    assert s == {"hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.75, "mrr": (1 + 0.5 + 0 + 0.2) / 4}


def test_evaluate_finds_expected_files(docs_dir, memory_store, embedder):
    ingest([str(docs_dir)], memory_store, embedder)
    path = {p.rsplit("/", 1)[-1]: p for p in memory_store.docs}
    questions = [
        {"question": "How do I clear the cache?", "source": path["caching.md"]},
        {"question": "How do I upgrade uv?", "source": path["install.md"]},
        {"question": "Which file pins the Python version?", "source": path["python.md"]},
    ]
    assert evaluate(questions, memory_store, embedder) == [1, 1, 1]
