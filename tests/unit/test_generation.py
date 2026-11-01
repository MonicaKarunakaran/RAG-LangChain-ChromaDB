from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag_app.generation.rag_chain import (
    format_documents,
    create_rag_chain,
)


def test_format_documents_includes_content_and_citations():
    documents = [
        Document(
            page_content="Krishna explains the importance of duty.",
            metadata={"source": "gita.pdf", "page": 10},
        ),
        Document(
            page_content="Perform your prescribed duties.",
            metadata={"source": "gita.pdf", "page": 11},
        ),
    ]

    result = format_documents(documents)

    assert "[Source: gita.pdf, Page: 10]" in result
    assert "Krishna explains the importance of duty." in result
    assert "[Source: gita.pdf, Page: 11]" in result
    assert "Perform your prescribed duties." in result


def test_format_documents_handles_missing_metadata():
    documents = [
        Document(
            page_content="Duty is important.",
            metadata={},
        )
    ]

    result = format_documents(documents)

    assert "[Source: Unknown, Page: Unknown]" in result
    assert "Duty is important." in result


def test_format_documents_with_empty_list():
    assert format_documents([]) == ""


@patch("src.rag_app.generation.rag_chain.create_llm")
def test_create_rag_chain_generates_answer(mock_create_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Perform your duty sincerely."
    )
    mock_create_llm.return_value = mock_llm

    rag_chain = create_rag_chain()

    documents = [
        Document(
            page_content="Perform your prescribed duty.",
            metadata={"source": "gita.pdf", "page": 5},
        )
    ]

    result = rag_chain(
        {
            "question": "What should a person do?",
            "documents": documents,
        }
    )

    assert result == "Perform your duty sincerely."
    mock_llm.invoke.assert_called_once()

    prompt = mock_llm.invoke.call_args.args[0]
    prompt_text = str(prompt)

    assert "Perform your prescribed duty." in prompt_text
    assert "What should a person do?" in prompt_text
    assert "gita.pdf" in prompt_text
    assert "Page: 5" in prompt_text
