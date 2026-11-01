from src.rag_app.config.settings import TOP_K
from src.rag_app.ingestion.embeddings import create_embedding_model
from src.rag_app.retrieval.retriever import create_retriever
from langchain_chroma import Chroma


def main() -> None:
    print("=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    embeddings = create_embedding_model()

    vector_store = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory="vector_store/chroma",
    )

    retriever = create_retriever(
        vector_store=vector_store,
        top_k=TOP_K,
    )

    question = "What is the importance of performing one's duty?"

    print(f"\nQuestion:\n{question}\n")

    documents = retriever.invoke(question)

    print(f"Retrieved documents: {len(documents)}")

    for index, document in enumerate(documents, start=1):
        print("\n" + "-" * 60)
        print(f"RESULT {index}")
        print("-" * 60)

        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page"))

        print("\nContent:")
        print(document.page_content[:1000])


if __name__ == "__main__":
    main()