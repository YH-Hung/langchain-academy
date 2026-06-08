"""Self-contained offline mocks for the module-4 studio graphs.

Kept local to the studio folder (mirrors the repo-root ``lc_local`` package) so
``langgraph dev`` needs no extra packages. Toggle with ``USE_MOCKS`` in this
folder's ``.env`` (``USE_MOCKS=true`` by default).
"""
import os

from langchain_core.documents import Document


def use_mocks() -> bool:
    return os.environ.get("USE_MOCKS", "true").lower() in ("1", "true", "yes", "on")


def _canned_passages(query: str, n: int) -> list[str]:
    return [
        f"This is mock search result #{i + 1} about '{query}'. It contains "
        "placeholder content so the graph can run fully offline. Set USE_MOCKS=false "
        "in your .env (and provide an API key) to use the real service."
        for i in range(n)
    ]


class MockTavilySearch:
    """Drop-in replacement for ``langchain_tavily.TavilySearch``."""

    def __init__(self, max_results: int = 3, **kwargs):
        self.max_results = max_results

    def invoke(self, input, *args, **kwargs):
        query = input.get("query", "") if isinstance(input, dict) else str(input)
        results = [
            {
                "title": f"Mock result {i + 1} for {query}",
                "url": f"https://example.com/mock/{i + 1}",
                "content": passage,
                "score": round(1.0 - i * 0.1, 2),
            }
            for i, passage in enumerate(_canned_passages(query, self.max_results))
        ]
        return {"query": query, "results": results}


class MockWikipediaLoader:
    """Drop-in replacement for ``langchain_community...WikipediaLoader``."""

    def __init__(self, query: str, load_max_docs: int = 2, **kwargs):
        self.query = query
        self.load_max_docs = load_max_docs

    def load(self) -> list[Document]:
        slug = self.query.replace(" ", "_")
        return [
            Document(
                page_content=passage,
                metadata={
                    "source": f"https://en.wikipedia.org/wiki/{slug}",
                    "title": f"{self.query} (mock article {i + 1})",
                    "summary": f"Mock Wikipedia summary for '{self.query}'.",
                },
            )
            for i, passage in enumerate(_canned_passages(self.query, self.load_max_docs))
        ]


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
