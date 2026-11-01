import os

import pytest

from src.rag_app.generation.llm import create_llm


def test_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError):
        create_llm()