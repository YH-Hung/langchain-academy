"""Offline, deterministic mocks for the external search APIs used in the course.

These are drop-in replacements for ``langchain_tavily.TavilySearch`` and
``langchain_community.document_loaders.WikipediaLoader``. They return canned data
so notebooks and studio graphs run with no network access and no API keys.
"""
from langchain_core.documents import Document


def canned_passages(query: str, n: int) -> list[str]:
    """Deterministic placeholder passages keyed off the query."""
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
            for i, passage in enumerate(canned_passages(query, self.max_results))
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
            for i, passage in enumerate(canned_passages(self.query, self.load_max_docs))
        ]
