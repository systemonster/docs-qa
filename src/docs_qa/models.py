from pydantic import BaseModel, Field


class Hit(BaseModel):
    """One retrieved chunk. `score` is cosine similarity (higher is better)."""

    chunk_id: int
    file: str
    heading: str
    text: str
    score: float


class Citation(BaseModel):
    n: int
    file: str
    heading: str
    score: float
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


def to_citations(hits: list[Hit]) -> list[Citation]:
    return [
        Citation(n=i, file=h.file, heading=h.heading, score=round(h.score, 4), snippet=h.text[:240])
        for i, h in enumerate(hits, start=1)
    ]
