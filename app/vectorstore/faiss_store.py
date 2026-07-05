"""FAISS vector-store creation and persistence."""

from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.tools.embeddings import get_embedding_model


VECTOR_STORE_PATH = Path("vector_store/medicine_faiss")


def save_vector_store(
    chunks: Sequence[Document],
    path: Path = VECTOR_STORE_PATH,
) -> None:
    """Create a FAISS index from document chunks and save it locally."""

    if not chunks:
        raise ValueError("No chunks available. Cannot create vector store.")

    print(f"[INFO] Creating embeddings for {len(chunks)} chunks...")

    vector_store = FAISS.from_documents(
        documents=list(chunks),
        embedding=get_embedding_model(),
    )

    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))

    print(f"[SUCCESS] FAISS vector store saved at: {path}")
