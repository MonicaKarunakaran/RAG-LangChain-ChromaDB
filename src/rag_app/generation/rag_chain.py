from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.rag_app.generation.llm import create_llm
from src.rag_app.generation.prompts import RAG_PROMPT


def format_documents(documents: list[Document]) -> str:
    """Format retrieved documents into LLM context."""

    formatted_documents = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown",
        )
        page = document.metadata.get(
            "page",
            "Unknown",
        )

        formatted_documents.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_documents)


def create_rag_chain():
    """Create the LangChain RAG generation chain."""

    llm = create_llm()

    def generate_answer(inputs: dict) -> str:
        documents = inputs["documents"]
        question = inputs["question"]

        context = format_documents(documents)

        prompt = RAG_PROMPT.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        response = llm.invoke(prompt)

        return StrOutputParser().invoke(response)

    return generate_answer