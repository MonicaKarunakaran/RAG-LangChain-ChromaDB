from pathlib import Path

from src.rag_app.pipeline import ingest_document


def main() -> None:
    pdf_files = list(Path("data/raw").glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in data/raw."
        )

    pdf_path = pdf_files[0]

    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    ingest_document(pdf_path)

    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()