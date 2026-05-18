"""World news and macro context search helper.

This tool is for context wider than ticker-specific finance news: geopolitics,
macro events, regulation, supply chain issues, and sector trends.
"""

from __future__ import annotations

import time
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException


def search_world_news(
    query: str,
    limit: int = 5,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Search the web with DuckDuckGo and return compact structured results."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must not be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        results: list[dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(cleaned_query, max_results=limit):
                    results.append(
                        {
                            "query": cleaned_query,
                            "title": item.get("title"),
                            "snippet": item.get("body"),
                            "link": item.get("href"),
                            "source": item.get("source"),
                            "error": None,
                        }
                    )
        except DDGSException as exc:
            last_error = str(exc)
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
            continue

        if results:
            return results

        last_error = "No results returned."
        if attempt < max_attempts:
            time.sleep(retry_delay_seconds)

    return [
        {
            "query": cleaned_query,
            "title": None,
            "snippet": None,
            "link": None,
            "source": "ddgs",
            "error": last_error,
        }
    ]


def format_world_news_for_agent(query: str, limit: int = 5) -> str:
    """Format search results into compact text for an analyst agent prompt."""
    results = search_world_news(query, limit=limit)
    return format_world_news_results(query, results)


def format_world_news_results(query: str, results: list[dict[str, Any]]) -> str:
    """Format already-fetched search results for an analyst agent prompt."""
    if not results:
        return f"No search results found for query: {query}"

    lines = [f"Web search results for: {query}"]
    for idx, result in enumerate(results, start=1):
        if result.get("error"):
            lines.append(f"{idx}. Search error")
            lines.append(f"   Source: {result.get('source') or 'unknown'}")
            lines.append(f"   Error: {result['error']}")
            continue

        title = result.get("title") or "Untitled"
        snippet = result.get("snippet") or ""
        link = result.get("link") or "No link"

        lines.append(f"{idx}. {title}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        lines.append(f"   Link: {link}")

    return "\n".join(lines)


# Backward-compatible aliases while the project is being renamed.
search_web = search_world_news
format_search_for_agent = format_world_news_for_agent
format_search_results = format_world_news_results
