"""Finance news retrieval helpers for market analysis.

This module is for ticker-specific financial news. The first implementation
uses Yahoo Finance RSS because it is free and does not require an API key.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import feedparser


def _normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker must not be empty")
    return cleaned


def get_yahoo_rss_news(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent Yahoo Finance RSS headlines for a ticker."""
    symbol = _normalize_ticker(ticker)
    params = urlencode({"s": symbol, "region": "US", "lang": "en-US"})
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?{params}"

    feed = feedparser.parse(url)
    if getattr(feed, "bozo", False):
        raise ValueError(f"Could not parse Yahoo Finance RSS feed for {symbol}")

    articles: list[dict[str, Any]] = []
    for entry in feed.entries[:limit]:
        articles.append(
            {
                "ticker": symbol,
                "title": entry.get("title"),
                "publisher": entry.get("source", {}).get("title"),
                "published": entry.get("published"),
                "link": entry.get("link"),
                "summary": entry.get("summary"),
            }
        )

    return articles


def format_news_for_agent(ticker: str, limit: int = 10) -> str:
    """Format recent news into compact text for an analyst agent prompt."""
    articles = get_yahoo_rss_news(ticker, limit=limit)
    if not articles:
        return f"No recent Yahoo Finance RSS news found for {ticker.upper()}."

    lines = [f"Recent news for {ticker.upper()}:"]
    for idx, article in enumerate(articles, start=1):
        title = article.get("title") or "Untitled"
        publisher = article.get("publisher") or "Unknown source"
        published = article.get("published") or "Unknown date"
        summary = article.get("summary") or ""

        lines.append(f"{idx}. {title}")
        lines.append(f"   Source: {publisher} | Published: {published}")
        if summary:
            lines.append(f"   Summary: {summary}")

    return "\n".join(lines)
