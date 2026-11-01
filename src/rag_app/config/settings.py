import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    "vector_store/chroma",
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)

RELEVANCE_THRESHOLD = float(
    os.getenv("RELEVANCE_THRESHOLD", "0.45")
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)


def validate_settings() -> None:
    """Validate required application settings."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it to the .env file."
        )