from langchain_chroma import Chroma

from src.rag_app.config.settings import (
    TOP_K,
    RELEVANCE_THRESHOLD,
    CHROMA_PERSIST_DIRECTORY,
)
from src.rag_app.generation.rag_chain import (
    create_rag_chain,
)
from src.rag_app.ingestion.embeddings import (
    create_embedding_model,
)


class RAGService:
    """Application service for document-based question answering."""

    def __init__(self) -> None:
        self.embeddings = create_embedding_model()

        self.vector_store = Chroma(
            collection_name="rag_documents",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )

        self.rag_chain = create_rag_chain()

    def ask(self, question: str) -> dict:
        """Answer a question using relevant document context."""

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # Retrieve documents using raw similarity distances.
        results = (
            self.vector_store
            .similarity_search_with_score(
                question,
                k=TOP_K,
            )
        )

        # Chroma returns distances where lower means
        # more similar. Convert distance to a simple
        # similarity score between 0 and 1.
        relevant_documents = []

        for document, distance in results:
            similarity_score = 1 / (1 + distance)

            if similarity_score >= RELEVANCE_THRESHOLD:
                relevant_documents.append(document)

        # Reject questions unrelated to the document.
        if not relevant_documents:
            return {
                "answer": (
                    "I couldn't find the answer in "
                    "the provided document."
                ),
                "sources": [],
            }

        # Generate answer using only relevant documents.
        answer = self.rag_chain(
            {
                "question": question,
                "documents": relevant_documents,
            }
        )

        # Generate citations from the documents
        # actually passed to the LLM.
        sources = [
            {
                "source": document.metadata.get(
                    "source",
                    "Unknown",
                ),
                "page": document.metadata.get(
                    "page",
                    "Unknown",
                ),
            }
            for document in relevant_documents
        ]

        return {
            "answer": answer,
            "sources": sources,
        }