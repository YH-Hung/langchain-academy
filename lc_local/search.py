"""Search helpers that return mock or real implementations based on ``USE_MOCKS``."""
import os

from .mocks import MockTavilySearch, MockWikipediaLoader


def use_mocks() -> bool:
    return os.environ.get("USE_MOCKS", "true").lower() in ("1", "true", "yes", "on")


def web_search(max_results: int = 3):
    """Tavily-compatible web search tool (mock by default, real when USE_MOCKS is off)."""
    if use_mocks():
        return MockTavilySearch(max_results=max_results)
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=max_results)


def wikipedia_loader(query: str, load_max_docs: int = 2):
    """WikipediaLoader-compatible loader (mock by default, real when USE_MOCKS is off)."""
    if use_mocks():
        return MockWikipediaLoader(query=query, load_max_docs=load_max_docs)
    from langchain_community.document_loaders import WikipediaLoader

    return WikipediaLoader(query=query, load_max_docs=load_max_docs)
