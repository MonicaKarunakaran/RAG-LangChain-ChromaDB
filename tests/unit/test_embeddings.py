from src.rag_app.ingestion.embeddings import create_embedding_model


def test_embedding_model_creation():
    embeddings = create_embedding_model()

    assert embeddings is not None


def test_embedding_generation():
    embeddings = create_embedding_model()

    vector = embeddings.embed_query(
        "What is the purpose of performing one's duty?"
    )

    assert isinstance(vector, list)
    assert len(vector) > 0