from pathlib import Path

from src.rag_app.pipeline import ingest_document


def test_ingestion_pipeline(tmp_path):
    pdf_files = list(Path("data/raw").glob("*.pdf"))

    assert pdf_files, "No PDF found in data/raw."

    vector_store = ingest_document(
        file_path=pdf_files[0],
        collection_name="integration_test",
    )

    assert vector_store is not None