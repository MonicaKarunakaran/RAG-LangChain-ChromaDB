from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_app.config.settings import CHROMA_PERSIST_DIRECTORY


def create_vector_store(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str = "rag_documents",
    persist_directory: str | None = None,
) -> Chroma:
    """
    Create or update a persistent ChromaDB vector store.
    """

    if not documents:
        raise ValueError(
            "Cannot create vector store from empty documents."
        )

    directory = Path(
        persist_directory or CHROMA_PERSIST_DIRECTORY
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(directory),
    )

    return vector_store