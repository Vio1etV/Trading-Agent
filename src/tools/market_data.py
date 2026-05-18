"""Market data helpers backed by yfinance.

These functions are intentionally small and importable. Agent code should call
them as tools, then decide how to summarize the returned data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


def _normalize_ticker(ticker: str) -> str:
    """Return a clean ticker symbol for yfinance calls."""
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker must not be empty")
    return cleaned


def get_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker.

    Args:
        ticker: Stock ticker, for example "NVDA".
        period: yfinance period string, for example "1mo", "6mo", "1y".
        interval: yfinance interval string, for example "1d", "1h".

    Returns:
        DataFrame with Date plus OHLCV columns.
    """
    symbol = _normalize_ticker(ticker)
    data = yf.Ticker(symbol).history(period=period, interval=interval)

    if data.empty:
        raise ValueError(f"No price history returned for ticker {symbol}")

    data = data.reset_index()
    data.insert(0, "Ticker", symbol)
    return data


def get_latest_price(ticker: str) -> dict[str, Any]:
    """Fetch a compact latest-price snapshot for a ticker."""
    symbol = _normalize_ticker(ticker)
    history = yf.Ticker(symbol).history(period="5d", interval="1d")

    if history.empty:
        raise ValueError(f"No recent price data returned for ticker {symbol}")

    latest = history.iloc[-1]
    latest_date = history.index[-1]

    return {
        "ticker": symbol,
        "date": str(latest_date.date()),
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": int(latest["Volume"]),
    }


def get_company_info(ticker: str) -> dict[str, Any]:
    """Fetch a small company profile from yfinance."""
    symbol = _normalize_ticker(ticker)
    info = yf.Ticker(symbol).get_info()

    if not info:
        raise ValueError(f"No company info returned for ticker {symbol}")

    fields = [
        "longName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
        "currency",
    ]

    profile = {"ticker": symbol}
    for field in fields:
        profile[field] = info.get(field)

    return profile
