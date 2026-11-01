from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF and convert each page into a LangChain Document.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of LangChain Document objects.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {path.suffix}"
        )

    reader = PdfReader(str(path))

    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "page": page_number,
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No extractable text found in: {path}"
        )

    return documents