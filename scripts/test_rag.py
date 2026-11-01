from langchain_chroma import Chroma

from src.rag_app.config.settings import TOP_K
from src.rag_app.generation.rag_chain import (
    create_rag_chain,
    retrieve_sources,
)
from src.rag_app.ingestion.embeddings import create_embedding_model
from src.rag_app.retrieval.retriever import create_retriever


def main() -> None:
    print("=" * 60)
    print("COMPLETE RAG PIPELINE TEST")
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

    rag_chain = create_rag_chain(retriever)

    question = "What is the importance of performing one's duty?"

    print(f"\nQuestion:\n{question}")

    answer = rag_chain.invoke(question)

    sources = retrieve_sources(
        retriever,
        question,
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, source in enumerate(sources, start=1):
        print(f"\n{index}.")
        print(f"   Source: {source['source']}")
        print(f"   Page: {source['page']}")


if __name__ == "__main__":
    main()