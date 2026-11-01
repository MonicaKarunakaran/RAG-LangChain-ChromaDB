from pathlib import Path

from src.rag_app.ingestion.chunking import split_documents
from src.rag_app.ingestion.loaders import load_pdf


def main() -> None:
    pdf_files = list(Path("data/raw").glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF found in data/raw"
        )

    pdf_path = pdf_files[0]

    print(f"Loading: {pdf_path}")

    documents = load_pdf(pdf_path)

    print(f"Pages loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    print("\nSample chunk:\n")
    print(chunks[0].page_content[:1000])

    print("\nMetadata:")
    print(chunks[0].metadata)


if __name__ == "__main__":
    main()