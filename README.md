# RAG Pipeline with LangChain & ChromaDB

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, Hugging Face embeddings, and the Groq API. It retrieves relevant information from the *Bhagavad Gita – As It Is* PDF and uses the retrieved context to generate answers with source citations.

The project includes a command-line interface, a Streamlit web interface, and automated tests.

## Features

- PDF document loading and text extraction
- Text chunking for efficient retrieval
- Semantic search using Hugging Face embeddings
- Persistent vector storage using ChromaDB
- Context-based answer generation using the Groq API
- Source and page citations
- Relevance filtering for retrieved documents
- Fallback response when relevant information is not found
- Command-line and Streamlit interfaces
- Unit and integration tests

## Technology Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Groq API
- Streamlit
- Pytest
- Docker

## Project Structure

```text
RAG-LangChain-ChromaDB/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
├── scripts/
│   ├── ingest.py
│   ├── evaluate.py
│   └── test_*.py
├── src/
│   └── rag_app/
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-lock.txt
├── pytest.ini
└── run.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/MonicaKarunakaran/RAG-LangChain-ChromaDB.git
cd RAG-LangChain-ChromaDB
```

### 2. Create and activate a virtual environment

On Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Add your own Groq API key to `.env`. Never commit your actual API key.

### 5. Ingest the PDF

Run the ingestion script:

```bash
python scripts/ingest.py
```

Run ingestion if you need to build or rebuild the vector database.

## Running the Application

### Command-line interface

```bash
python run.py
```

### Streamlit interface

```bash
streamlit run app/streamlit_app.py
```

## Running Tests

Run the complete test suite:

```bash
pytest -v
```

The current test suite has **23 passing tests**.

## Configuration

The application uses environment variables to configure the model, embedding model, vector database location, retrieval count, relevance threshold, and chunking settings. See `.env.example` for the configuration template.

## Security

- Keep API keys in `.env`.
- Do not commit `.env` or other files containing secrets.
- Use `.env.example` with placeholder values.
- Review the source PDF's copyright and distribution terms before redistributing it.

## Future Improvements

- Improve retrieval quality through evaluation and parameter tuning.
- Add more document formats.
- Improve answer evaluation and monitoring.
- Deploy the application as a cloud-hosted service.