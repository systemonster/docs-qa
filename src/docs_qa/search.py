from docs_qa.db import Store
from docs_qa.embeddings import Embedder
from docs_qa.models import Hit


def retrieve(store: Store, embedder: Embedder, query: str, k: int) -> list[Hit]:
    """Top-k chunks by cosine similarity to the query."""
    return store.search(embedder.embed_query(query), k)
