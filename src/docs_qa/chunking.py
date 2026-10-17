"""Heading-aware Markdown chunking.

Sections are cut at headings of level 1-3 (deeper headings stay inside their parent).
A section longer than `max_words` is split on line boundaries into windows that share
about `overlap` words, so a sentence cut at a boundary still appears whole in one window.
"""

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_MIN_WORDS = 5


@dataclass(frozen=True)
class Chunk:
    heading: str  # breadcrumb, e.g. "Caching > Cache types"
    text: str
    ordinal: int

    @property
    def embed_text(self) -> str:
        """What gets embedded: the breadcrumb gives the model context the body may lack."""
        return f"{self.heading}\n\n{self.text}"


def _words(line: str) -> int:
    return len(line.split())


def split_sections(markdown: str) -> list[tuple[str, list[str]]]:
    """Return (breadcrumb, body lines) per section. Headings inside code fences are ignored."""
    path: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    body: list[str] = []
    crumb = ""
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in body):
            sections.append((crumb, body.copy()))
        body.clear()

    for line in _FRONTMATTER.sub("", markdown.replace("\r\n", "\n")).split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2)
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            crumb = " > ".join(t for _, t in path)
        else:
            body.append(line)
    flush()
    return sections


def window(lines: list[str], max_words: int, overlap: int) -> list[str]:
    """Greedy line-based windows of at most ~max_words; neighbours share ~overlap words."""
    out: list[str] = []
    cur: list[str] = []
    count = 0
    for line in lines:
        w = _words(line)
        if cur and count + w > max_words:
            out.append("\n".join(cur).strip())
            tail: list[str] = []
            tail_words = 0
            for prev in reversed(cur):
                tail_words += _words(prev)
                if tail_words > overlap:
                    break
                tail.insert(0, prev)
            cur = tail
            count = sum(_words(x) for x in cur)
        cur.append(line)
        count += w
    if cur:
        out.append("\n".join(cur).strip())
    return [c for c in out if c]


def chunk_markdown(markdown: str, max_words: int = 400, overlap: int = 60) -> list[Chunk]:
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    chunks: list[Chunk] = []
    for crumb, lines in split_sections(markdown):
        for text in window(lines, max_words, overlap):
            if _words(text) >= _MIN_WORDS:
                chunks.append(Chunk(crumb or "(intro)", text, len(chunks)))
    return chunks
