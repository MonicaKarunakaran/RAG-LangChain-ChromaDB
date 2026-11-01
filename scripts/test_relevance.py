from langchain_chroma import Chroma

from src.rag_app.ingestion.embeddings import create_embedding_model


def main():
    print("=" * 60)
    print("RAG RELEVANCE SCORE TEST")
    print("=" * 60)

    embeddings = create_embedding_model()

    vector_store = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory="vector_store/chroma",
    )

    questions = [
        "What is the importance of performing one's duty?",
        "What does the Bhagavad Gita say about controlling the mind?",
        "What is the capital of France?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        results = vector_store.similarity_search_with_relevance_scores(
            question,
            k=5,
        )

        if not results:
            print("No results found.")
            continue

        for index, (document, score) in enumerate(
            results,
            start=1,
        ):
            print(f"\nResult {index}")
            print(f"Score: {score:.4f}")
            print(
                f"Page: {document.metadata.get('page')}"
            )
            print(
                "Content: "
                + document.page_content[:200]
                .replace("\n", " ")
            )


if __name__ == "__main__":
    main()