from pathlib import Path

import pytest

from src.rag_app.ingestion.loaders import load_pdf


def test_load_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("data/raw/nonexistent.pdf")


def test_load_pdf_rejects_non_pdf():
    test_file = Path("data/raw/test.txt")
    test_file.write_text("test")

    try:
        with pytest.raises(ValueError):
            load_pdf(test_file)
    finally:
        test_file.unlink()


def test_load_real_pdf():
    pdf_files = list(Path("data/raw").glob("*.pdf"))

    assert pdf_files, "No PDF found in data/raw"

    documents = load_pdf(pdf_files[0])

    assert len(documents) > 0
    assert documents[0].page_content.strip()
    assert "source" in documents[0].metadata
    assert "page" in documents[0].metadata