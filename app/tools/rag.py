"""RAG search helpers."""

from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.tools.embeddings import get_embedding_model
from app.vectorstore.faiss_store import VECTOR_STORE_PATH


@lru_cache(maxsize=1)
def load_vector_store(path: str = str(VECTOR_STORE_PATH)) -> FAISS:
    """Load the persisted FAISS vector store."""

    return FAISS.load_local(
        folder_path=str(Path(path)),
        embeddings=get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def search_medicine_knowledge(query: str,k: int = 4,) -> list[Document]:
    """Search local medicine knowledge using FAISS."""

    if not query.strip():
        return []

    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)
