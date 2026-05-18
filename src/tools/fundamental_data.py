"""Fundamental data retrieval for company background and filings."""

from __future__ import annotations

from typing import Any

import requests
import yfinance as yf


USER_AGENT = "trading-agent-class-project/0.1 contact: duy0341@northwestern.edu"


def _normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker must not be empty")
    return cleaned


def get_company_profile(ticker: str) -> dict[str, Any]:
    """Fetch company profile and valuation fields from yfinance."""
    symbol = _normalize_ticker(ticker)
    info = yf.Ticker(symbol).get_info()

    if not info:
        raise ValueError(f"No company info returned for ticker {symbol}")

    fields = [
        "longName",
        "shortName",
        "sector",
        "industry",
        "longBusinessSummary",
        "marketCap",
        "enterpriseValue",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "profitMargins",
        "revenueGrowth",
        "earningsGrowth",
        "totalRevenue",
        "grossProfits",
        "ebitda",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
        "currency",
    ]

    profile = {"ticker": symbol}
    for field in fields:
        profile[field] = info.get(field)

    return profile


def get_sec_cik(ticker: str) -> str | None:
    """Resolve a ticker to a SEC CIK string."""
    symbol = _normalize_ticker(ticker)
    url = "https://www.sec.gov/files/company_tickers.json"

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None

    for item in payload.values():
        if item.get("ticker", "").upper() == symbol:
            return str(item.get("cik_str")).zfill(10)

    return None


def get_recent_sec_filings(ticker: str, limit: int = 10) -> dict[str, Any]:
    """Fetch recent SEC filing metadata for a ticker."""
    symbol = _normalize_ticker(ticker)
    cik = get_sec_cik(symbol)
    if cik is None:
        return {"ticker": symbol, "cik": None, "error": "CIK not found", "filings": []}

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"ticker": symbol, "cik": cik, "error": str(exc), "filings": []}

    recent = (payload.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    accession_numbers = recent.get("accessionNumber") or []
    primary_documents = recent.get("primaryDocument") or []

    filings = []
    for idx, form in enumerate(forms[:limit]):
        filings.append(
            {
                "form": form,
                "filing_date": filing_dates[idx] if idx < len(filing_dates) else None,
                "accession_number": (
                    accession_numbers[idx] if idx < len(accession_numbers) else None
                ),
                "primary_document": (
                    primary_documents[idx] if idx < len(primary_documents) else None
                ),
            }
        )

    return {"ticker": symbol, "cik": cik, "error": None, "filings": filings}


def get_fundamental_context(ticker: str) -> dict[str, Any]:
    """Fetch yfinance profile and recent SEC filing metadata."""
    symbol = _normalize_ticker(ticker)
    return {
        "ticker": symbol,
        "company_profile": get_company_profile(symbol),
        "recent_sec_filings": get_recent_sec_filings(symbol),
    }


def format_fundamental_for_agent(ticker: str) -> str:
    """Format fundamental data into compact text for an analyst agent prompt."""
    data = get_fundamental_context(ticker)
    profile = data["company_profile"]
    filings = data["recent_sec_filings"]

    lines = [f"Fundamental data for {ticker.upper()}:"]

    name = profile.get("longName") or profile.get("shortName") or ticker.upper()
    sector = profile.get("sector") or "N/A"
    industry = profile.get("industry") or "N/A"
    lines.append(f"Company: {name} | Sector: {sector} | Industry: {industry}")

    market_cap = profile.get("marketCap")
    if market_cap:
        lines.append(f"Market cap: ${market_cap:,.0f}")

    pe_trailing = profile.get("trailingPE")
    pe_forward = profile.get("forwardPE")
    pb = profile.get("priceToBook")
    pe_parts = []
    if pe_trailing is not None:
        pe_parts.append(f"trailing P/E={pe_trailing:.1f}")
    if pe_forward is not None:
        pe_parts.append(f"forward P/E={pe_forward:.1f}")
    if pb is not None:
        pe_parts.append(f"P/B={pb:.2f}")
    if pe_parts:
        lines.append("Valuation: " + ", ".join(pe_parts))

    low52 = profile.get("fiftyTwoWeekLow")
    high52 = profile.get("fiftyTwoWeekHigh")
    if low52 and high52:
        lines.append(f"52-week range: ${low52} – ${high52}")

    summary = (profile.get("longBusinessSummary") or "")[:300]
    if summary:
        lines.append(f"Business: {summary}")

    filing_list = filings.get("filings") or []
    if filing_list:
        lines.append("Recent SEC filings:")
        for f in filing_list[:5]:
            lines.append(f"  {f['form']} ({f['filing_date']})")
    elif filings.get("error"):
        lines.append(f"SEC filings: unavailable ({filings['error']})")

    return "\n".join(lines)
