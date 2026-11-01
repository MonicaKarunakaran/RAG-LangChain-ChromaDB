from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag_app.config.settings import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split documents into smaller chunks for retrieval.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """

    if not documents:
        raise ValueError("No documents provided for chunking.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Chunking produced no documents.")

    return chunks