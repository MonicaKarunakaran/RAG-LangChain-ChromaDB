from src.rag_app.config.settings import (
    CHROMA_PERSIST_DIRECTORY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    TOP_K,
)


def test_default_settings():
    assert GROQ_MODEL
    assert EMBEDDING_MODEL
    assert CHROMA_PERSIST_DIRECTORY
    assert TOP_K > 0
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0