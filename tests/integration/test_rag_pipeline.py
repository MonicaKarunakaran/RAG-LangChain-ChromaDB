from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag_app.application.rag_service import RAGService


FALLBACK_ANSWER = (
    "I couldn't find the answer in the provided document."
)


@pytest.fixture
def rag_service():
    with (
        patch(
            "src.rag_app.application.rag_service.create_embedding_model"
        ),
        patch(
            "src.rag_app.application.rag_service.Chroma"
        ) as mock_chroma,
        patch(
            "src.rag_app.application.rag_service.create_rag_chain"
        ) as mock_create_chain,
    ):
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        mock_chain = MagicMock(return_value="Duty is important.")
        mock_create_chain.return_value = mock_chain

        service = RAGService()

        yield service, mock_vector_store, mock_chain


def test_rag_pipeline_returns_answer_and_sources(rag_service):
    service, vector_store, chain = rag_service

    document = Document(
        page_content="A person should perform their duty.",
        metadata={"source": "gita.pdf", "page": 12},
    )

    vector_store.similarity_search_with_score.return_value = [
        (document, 0.2)
    ]

    result = service.ask("What should a person do?")

    assert result["answer"] == "Duty is important."
    assert result["sources"] == [
        {"source": "gita.pdf", "page": 12}
    ]
    chain.assert_called_once()


def test_rag_pipeline_rejects_irrelevant_question(rag_service):
    service, vector_store, chain = rag_service

    document = Document(
        page_content="An unrelated passage.",
        metadata={"source": "gita.pdf", "page": 3},
    )

    # This distance produces a score below the configured threshold.
    vector_store.similarity_search_with_score.return_value = [
        (document, 10.0)
    ]

    result = service.ask("What is the capital of France?")

    assert result["answer"] == FALLBACK_ANSWER
    assert result["sources"] == []
    chain.assert_not_called()


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_rag_pipeline_rejects_empty_question(rag_service, question):
    service, vector_store, chain = rag_service

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        service.ask(question)

    vector_store.similarity_search_with_score.assert_not_called()
    chain.assert_not_called()


def test_rag_pipeline_omits_sources_when_no_documents_match(
    rag_service,
):
    service, vector_store, chain = rag_service

    vector_store.similarity_search_with_score.return_value = []

    result = service.ask("Explain something not in the document.")

    assert result["answer"] == FALLBACK_ANSWER
    assert result["sources"] == []
    chain.assert_not_called()


def test_rag_pipeline_uses_only_relevant_documents(rag_service):
    service, vector_store, chain = rag_service

    relevant_document = Document(
        page_content="Perform your duty.",
        metadata={"source": "gita.pdf", "page": 5},
    )

    irrelevant_document = Document(
        page_content="An unrelated passage.",
        metadata={"source": "other.pdf", "page": 1},
    )

    vector_store.similarity_search_with_score.return_value = [
        (relevant_document, 0.2),
        (irrelevant_document, 10.0),
    ]

    service.ask("What is duty?")

    chain.assert_called_once()
    inputs = chain.call_args.args[0]

    assert inputs["question"] == "What is duty?"
    assert inputs["documents"] == [relevant_document]