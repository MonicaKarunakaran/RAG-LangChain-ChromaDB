from src.rag_app.application.rag_service import RAGService


def print_sources(sources: list[dict]) -> None:
    """Display source citations."""

    print("\nSources:")
    print("-" * 50)

    seen = set()

    for source in sources:
        source_name = source["source"]
        page = source["page"]

        citation = (source_name, page)

        if citation in seen:
            continue

        seen.add(citation)

        print(
            f"📄 {source_name} — Page {page}"
        )


def main() -> None:
    print("=" * 60)
    print("             RAG Q&A ASSISTANT")
    print("=" * 60)

    print("\nAsk questions about the document.")
    print("Type 'exit' or 'quit' to stop.\n")

    rag_service = RAGService()

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            result = rag_service.ask(question)

            print("\nAssistant:")
            print(result["answer"])

            print_sources(
                result["sources"]
            )

            print()

        except Exception as error:
            print(
                f"\nError: {error}\n"
            )


if __name__ == "__main__":
    main()