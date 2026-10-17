import pytest

from docs_qa.chunking import chunk_markdown, split_sections, window


def test_heading_breadcrumbs():
    md = "# Guide\n\nIntro text for the guide here.\n\n## Setup\n\nInstall the tool first please.\n"
    crumbs = [c.heading for c in chunk_markdown(md)]
    assert crumbs == ["Guide", "Guide > Setup"]


def test_sibling_heading_replaces_previous_in_breadcrumb():
    md = "# A\n\n## B\n\nbody of b with words.\n\n## C\n\nbody of c with words.\n"
    assert [c.heading for c in chunk_markdown(md)] == ["A > B", "A > C"]


def test_deep_headings_stay_inside_parent_section():
    md = "# A\n\nintro words intro words.\n\n#### Deep\n\ndetail words detail words.\n"
    chunks = chunk_markdown(md)
    assert len(chunks) == 1
    assert "#### Deep" in chunks[0].text


def test_headings_inside_code_fences_are_ignored():
    md = "# Real\n\n```bash\n# just a shell comment\nls -la\n```\n\nSome prose after the fence.\n"
    chunks = chunk_markdown(md)
    assert [c.heading for c in chunks] == ["Real"]
    assert "# just a shell comment" in chunks[0].text


def test_frontmatter_removed_and_heading_less_text_gets_intro_label():
    chunks = chunk_markdown("---\ntitle: x\n---\nplain text without any heading at all\n")
    assert [c.heading for c in chunks] == ["(intro)"]
    assert "title" not in chunks[0].text


def test_empty_sections_and_tiny_fragments_are_dropped():
    md = "# Empty\n\n## Also empty\n\n## Tiny\n\nfew words\n"
    assert chunk_markdown(md) == []


def test_long_section_is_windowed_with_overlap():
    lines = [f"line{i} " + " ".join(["w"] * 9) for i in range(60)]  # 10 words per line
    parts = window(lines, max_words=100, overlap=30)
    assert len(parts) > 1
    assert all(len(p.split()) <= 100 for p in parts)
    first_tail = parts[0].splitlines()[-1]
    assert first_tail in parts[1]  # overlap: last line of window 1 reappears in window 2
    assert {ln for p in parts for ln in p.splitlines()} == set(lines)  # nothing lost


def test_ordinals_are_sequential():
    para = "\n".join(["word word word word word"] * 20)
    md = f"# A\n\n{para}\n\n# B\n\n{para}"
    chunks = chunk_markdown(md, max_words=30, overlap=5)
    assert len(chunks) > 2
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_markdown("# A\n\ntext text text text text", max_words=10, overlap=10)


def test_split_sections_handles_crlf():
    sections = split_sections("# A\r\n\r\nbody text here now\r\n")
    assert sections == [("A", ["", "body text here now", ""])]


def test_embed_text_includes_breadcrumb():
    chunk = chunk_markdown("# A\n\n## B\n\nthis is the body text\n")[0]
    assert chunk.embed_text.startswith("A > B\n\n")
