import pytest

from docs_qa.embeddings import HashEmbedder
from docs_qa.ingest import display_path, find_markdown, fingerprint, ingest
from docs_qa.search import retrieve


def test_first_run_adds_second_run_changes_nothing(any_store, docs_dir, embedder):
    first = ingest([str(docs_dir)], any_store, embedder)
    assert (first.files, first.added, first.unchanged) == (3, 3, 0)
    count = any_store.count()
    assert count == first.chunks > 0

    again = ingest([str(docs_dir)], any_store, embedder)
    assert (again.added, again.updated, again.unchanged, again.chunks) == (0, 0, 3, 0)
    assert any_store.count() == count  # idempotent: no duplicates


def test_changed_file_replaces_its_chunks(any_store, docs_dir, embedder):
    ingest([str(docs_dir)], any_store, embedder)
    (docs_dir / "python.md").write_text(
        "# Python\n\nCompletely new text about interpreters here.\n"
    )
    report = ingest([str(docs_dir)], any_store, embedder)
    assert (report.updated, report.unchanged) == (1, 2)
    hit = retrieve(any_store, embedder, "completely new interpreters", 1)[0]
    assert hit.file.endswith("python.md") and "Completely new" in hit.text
    assert not any(
        "side by side" in h.text for h in retrieve(any_store, embedder, "interpreters", 10)
    )


def test_changing_embedder_or_chunk_size_triggers_reingest(any_store, docs_dir, embedder):
    ingest([str(docs_dir)], any_store, embedder)

    class Other(HashEmbedder):
        name = "hash-other"

    assert ingest([str(docs_dir)], any_store, Other()).updated == 3
    assert ingest([str(docs_dir)], any_store, Other(), max_words=200).updated == 3


def test_search_returns_best_file_first_with_scores(any_store, docs_dir, embedder):
    ingest([str(docs_dir)], any_store, embedder)
    hits = retrieve(any_store, embedder, "how to clear the cache", 3)
    assert hits[0].file.endswith("caching.md")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_find_markdown_accepts_files_and_folders_and_rejects_missing(docs_dir, tmp_path):
    single = docs_dir / "install.md"
    assert find_markdown([str(single)]) == [single]
    assert len(find_markdown([str(docs_dir)])) == 3
    with pytest.raises(FileNotFoundError):
        find_markdown([str(tmp_path / "nope")])


def test_display_path_is_relative_to_cwd(monkeypatch, docs_dir):
    monkeypatch.chdir(docs_dir.parent)
    assert display_path(docs_dir / "install.md") == "docs/install.md"


def test_fingerprint_depends_on_content_embedder_and_params(embedder):
    base = fingerprint(b"x", embedder, 400, 60)
    assert base == fingerprint(b"x", embedder, 400, 60)
    assert base != fingerprint(b"y", embedder, 400, 60)
    assert base != fingerprint(b"x", embedder, 300, 60)
