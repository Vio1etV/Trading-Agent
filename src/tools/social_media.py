"""Social media data retrieval for ticker-specific market discussion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import requests


USER_AGENT = "trading-agent-class-project/0.1"


def _normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker must not be empty")
    return cleaned


def get_stocktwits_messages(ticker: str, limit: int = 30) -> dict[str, Any]:
    """Fetch recent StockTwits messages for a ticker."""
    symbol = _normalize_ticker(ticker)
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"ticker": symbol, "source": "stocktwits", "error": str(exc), "messages": []}

    messages = []
    bullish = bearish = unlabeled = 0
    for item in payload.get("messages", [])[:limit]:
        sentiment_obj = (item.get("entities") or {}).get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        if sentiment == "Bullish":
            bullish += 1
        elif sentiment == "Bearish":
            bearish += 1
        else:
            unlabeled += 1

        messages.append(
            {
                "created_at": item.get("created_at"),
                "user": (item.get("user") or {}).get("username"),
                "sentiment": sentiment,
                "body": item.get("body"),
            }
        )

    return {
        "ticker": symbol,
        "source": "stocktwits",
        "error": None,
        "summary": {
            "bullish": bullish,
            "bearish": bearish,
            "unlabeled": unlabeled,
            "total": len(messages),
        },
        "messages": messages,
    }


def get_reddit_posts(
    ticker: str,
    subreddits: tuple[str, ...] = ("wallstreetbets", "stocks", "investing"),
    limit_per_subreddit: int = 5,
) -> dict[str, Any]:
    """Fetch recent Reddit posts mentioning a ticker from public JSON endpoints."""
    symbol = _normalize_ticker(ticker)
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for subreddit in subreddits:
        params = urlencode(
            {
                "q": symbol,
                "restrict_sr": "on",
                "sort": "new",
                "t": "week",
                "limit": limit_per_subreddit,
            }
        )
        url = f"https://www.reddit.com/r/{subreddit}/search.json?{params}"
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            errors.append(f"r/{subreddit}: {exc}")
            continue

        children = (payload.get("data") or {}).get("children") or []
        for child in children:
            data = child.get("data") or {}
            created_utc = data.get("created_utc")
            created_at = None
            if created_utc:
                created_at = datetime.fromtimestamp(
                    created_utc,
                    tz=timezone.utc,
                ).isoformat()

            body = (data.get("selftext") or "")[:200] or None
            results.append(
                {
                    "subreddit": subreddit,
                    "created_at": created_at,
                    "title": data.get("title"),
                    "score": data.get("score"),
                    "num_comments": data.get("num_comments"),
                    "url": data.get("url"),
                    "body": body,
                }
            )

    return {
        "ticker": symbol,
        "source": "reddit",
        "errors": errors,
        "posts": results,
    }


def get_social_media_context(ticker: str) -> dict[str, Any]:
    """Fetch StockTwits and Reddit context for a ticker."""
    return {
        "ticker": _normalize_ticker(ticker),
        "stocktwits": get_stocktwits_messages(ticker),
        "reddit": get_reddit_posts(ticker),
    }


def format_social_for_agent(ticker: str) -> str:
    """Format social media data into compact text for an analyst agent prompt."""
    data = get_social_media_context(ticker)
    lines = [f"Social media context for {ticker.upper()}:"]

    st = data["stocktwits"]
    if st.get("error"):
        lines.append(f"StockTwits: unavailable ({st['error']})")
    else:
        s = st["summary"]
        lines.append(
            f"StockTwits ({s['total']} messages): "
            f"bullish={s['bullish']}, bearish={s['bearish']}, neutral={s['unlabeled']}"
        )
        for msg in st["messages"][:5]:
            sentiment = msg.get("sentiment") or "?"
            body = (msg.get("body") or "")[:150]
            lines.append(f"  [{sentiment}] {body}")

    rd = data["reddit"]
    if rd.get("errors"):
        lines.append(f"Reddit errors: {rd['errors']}")
    posts = rd.get("posts", [])
    if posts:
        lines.append(f"Reddit ({len(posts)} posts):")
        for post in posts[:5]:
            title = post.get("title") or ""
            score = post.get("score") or 0
            body = post.get("body") or ""
            lines.append(f"  r/{post['subreddit']} | score={score} | {title}")
            if body:
                lines.append(f"    {body}")
    else:
        lines.append("Reddit: no posts found.")

    return "\n".join(lines)
