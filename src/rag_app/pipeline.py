from pathlib import Path

from langchain_chroma import Chroma

from src.rag_app.ingestion.chunking import split_documents
from src.rag_app.ingestion.embeddings import create_embedding_model
from src.rag_app.ingestion.loaders import load_pdf
from src.rag_app.retrieval.vector_store import create_vector_store


def ingest_document(
    file_path: str | Path,
    collection_name: str = "rag_documents",
) -> Chroma:
    """
    Run the complete document ingestion pipeline.

    Pipeline:
        PDF → Documents → Chunks → Embeddings → ChromaDB
    """

    print(f"Loading document: {file_path}")

    documents = load_pdf(file_path)

    print(f"Loaded {len(documents)} document pages.")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = create_embedding_model()

    print("Embedding model initialized.")

    vector_store = create_vector_store(
        documents=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
    )

    print("Documents successfully stored in ChromaDB.")

    return vector_store