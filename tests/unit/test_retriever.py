from langchain_core.documents import Document

import pytest

from src.rag_app.ingestion.embeddings import create_embedding_model
from src.rag_app.retrieval.vector_store import create_vector_store
from src.rag_app.retrieval.retriever import create_retriever


def test_create_vector_store(tmp_path):
    documents = [
        Document(
            page_content="Krishna explains the importance of duty.",
            metadata={
                "source": "test.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="One should perform one's prescribed duties.",
            metadata={
                "source": "test.pdf",
                "page": 2,
            },
        ),
    ]

    embeddings = create_embedding_model()

    vector_store = create_vector_store(
        documents=documents,
        embeddings=embeddings,
        collection_name="test_collection",
    )

    assert vector_store is not None


def test_create_vector_store_empty_documents():
    embeddings = create_embedding_model()

    with pytest.raises(ValueError):
        create_vector_store(
            documents=[],
            embeddings=embeddings,
        )