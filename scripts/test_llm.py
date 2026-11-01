from src.rag_app.generation.llm import create_llm


def main() -> None:
    print("=" * 60)
    print("GROQ LLM TEST")
    print("=" * 60)

    llm = create_llm()

    response = llm.invoke(
        "Explain what RAG means in one sentence."
    )

    print("\nResponse:")
    print(response.content)


if __name__ == "__main__":
    main()