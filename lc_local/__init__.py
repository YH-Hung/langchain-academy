"""Helpers to run LangChain Academy against a local OpenAI-compatible LLM and mocked APIs.

Usage in a notebook::

    from lc_local import setup_env, chat_model, web_search, wikipedia_loader

    setup_env()
    llm = chat_model(temperature=0)
"""
from .config import chat_model, setup_env
from .search import web_search, wikipedia_loader

__all__ = ["setup_env", "chat_model", "web_search", "wikipedia_loader"]
