from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_retriever(
    vector_store: Chroma,
    top_k: int = 5,
):
    """
    Create a similarity-based retriever.

    Args:
        vector_store: ChromaDB vector store.
        top_k: Number of documents to retrieve.

    Returns:
        LangChain retriever.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    return vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": top_k,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    },
)