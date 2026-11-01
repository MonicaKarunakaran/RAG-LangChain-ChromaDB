from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_app.config.settings import EMBEDDING_MODEL


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the embedding model used for document
    and query vector generation.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )