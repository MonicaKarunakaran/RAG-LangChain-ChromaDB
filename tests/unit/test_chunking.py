from langchain_core.documents import Document

import pytest

from src.rag_app.ingestion.chunking import split_documents


def test_split_documents():
    document = Document(
        page_content="This is a test document. " * 100,
        metadata={
            "source": "test.pdf",
            "page": 0,
        },
    )

    chunks = split_documents([document])

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 0


def test_split_documents_empty_input():
    with pytest.raises(ValueError):
        split_documents([])