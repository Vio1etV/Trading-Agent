"""Short-horizon technical indicators for risk analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from src.tools.market_data import get_latest_price, get_price_history


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pct_change(current: float, previous: float) -> float | None:
    if previous == 0 or pd.isna(previous):
        return None
    return float((current - previous) / previous)


def _trend_from_returns(return_value: float | None, threshold: float = 0.01) -> str:
    if return_value is None:
        return "unknown"
    if return_value > threshold:
        return "uptrend"
    if return_value < -threshold:
        return "downtrend"
    return "sideways"


def _position_in_range(price: float, low: float, high: float) -> str:
    if high <= low:
        return "unknown"
    position = (price - low) / (high - low)
    if position >= 0.8:
        return "near_high"
    if position <= 0.2:
        return "near_low"
    return "middle"


def compute_short_horizon_indicators(ticker: str) -> dict[str, Any]:
    """Compute technical signals focused on next-24h and next-1w decisions."""
    latest = get_latest_price(ticker)
    history = get_price_history(ticker, period="6mo", interval="1d")

    close = history["Close"].astype(float)
    high = history["High"].astype(float)
    low = history["Low"].astype(float)
    open_ = history["Open"].astype(float)
    volume = history["Volume"].astype(float)

    current_price = float(close.iloc[-1])

    previous_close = float(close.iloc[-2]) if len(close) >= 2 else np.nan
    close_5d_ago = float(close.iloc[-6]) if len(close) >= 6 else np.nan
    close_20d_ago = float(close.iloc[-21]) if len(close) >= 21 else np.nan

    return_1d = _pct_change(current_price, previous_close)
    return_5d = _pct_change(current_price, close_5d_ago)
    return_20d = _pct_change(current_price, close_20d_ago)

    last_open = float(open_.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    intraday_range = (last_high - last_low) / current_price if current_price else None
    close_vs_open = _pct_change(current_price, last_open)

    last_5 = history.tail(5)
    high_5d = float(last_5["High"].max())
    low_5d = float(last_5["Low"].min())
    support_5d = low_5d
    resistance_5d = high_5d
    position_5d = _position_in_range(current_price, low_5d, high_5d)

    avg_volume_5d = float(volume.tail(5).mean())
    volume_vs_5d_avg = float(volume.iloc[-1] / avg_volume_5d) if avg_volume_5d else None

    daily_returns = close.pct_change()
    volatility_5d = _safe_float(daily_returns.tail(5).std())
    volatility_20d = _safe_float(daily_returns.tail(20).std())

    rsi = RSIIndicator(close=close, window=14).rsi()
    macd_calc = MACD(close=close)
    macd_value = macd_calc.macd()
    macd_signal = macd_calc.macd_signal()
    bollinger = BollingerBands(close=close, window=20, window_dev=2)
    bollinger_upper = bollinger.bollinger_hband()
    bollinger_lower = bollinger.bollinger_lband()

    latest_macd = macd_value.iloc[-1]
    latest_macd_signal = macd_signal.iloc[-1]
    if pd.isna(latest_macd) or pd.isna(latest_macd_signal):
        macd_state = "unknown"
    elif latest_macd > latest_macd_signal:
        macd_state = "bullish"
    elif latest_macd < latest_macd_signal:
        macd_state = "bearish"
    else:
        macd_state = "neutral"

    upper = bollinger_upper.iloc[-1]
    lower = bollinger_lower.iloc[-1]
    bollinger_position = (
        _position_in_range(current_price, float(lower), float(upper))
        if not pd.isna(upper) and not pd.isna(lower)
        else "unknown"
    )

    return {
        "ticker": latest["ticker"],
        "current_price": current_price,
        "latest_price_snapshot": latest,
        "next_24h_signals": {
            "return_1d": return_1d,
            "intraday_range": intraday_range,
            "close_vs_open": close_vs_open,
            "volume_vs_5d_avg": volume_vs_5d_avg,
            "position_in_5d_range": position_5d,
        },
        "next_1w_signals": {
            "return_5d": return_5d,
            "trend_5d": _trend_from_returns(return_5d),
            "high_5d": high_5d,
            "low_5d": low_5d,
            "support_5d": support_5d,
            "resistance_5d": resistance_5d,
            "volatility_5d": volatility_5d,
        },
        "technical_context": {
            "return_20d": return_20d,
            "trend_20d": _trend_from_returns(return_20d),
            "volatility_20d": volatility_20d,
            "rsi_14": _safe_float(rsi.iloc[-1]),
            "macd_signal": macd_state,
            "macd": _safe_float(latest_macd),
            "macd_signal_value": _safe_float(latest_macd_signal),
            "bollinger_position": bollinger_position,
            "bollinger_upper": _safe_float(upper),
            "bollinger_lower": _safe_float(lower),
        },
    }


def format_indicators_for_agent(ticker: str) -> str:
    """Format technical indicators into compact text for a risk manager agent prompt."""
    data = compute_short_horizon_indicators(ticker)
    lines = [f"Technical indicators for {ticker.upper()}:"]

    price = data["current_price"]
    lines.append(f"Current price: ${price:.2f}")

    n24 = data["next_24h_signals"]
    r1d = n24.get("return_1d")
    vol_ratio = n24.get("volume_vs_5d_avg")
    position = n24.get("position_in_5d_range") or "unknown"
    r1d_str = f"{r1d:.2%}" if r1d is not None else "N/A"
    vol_str = f"{vol_ratio:.2f}x" if vol_ratio is not None else "N/A"
    lines.append(
        f"24h signals: return={r1d_str}, volume_vs_5d_avg={vol_str}, "
        f"position_in_5d_range={position}"
    )

    n1w = data["next_1w_signals"]
    r5d = n1w.get("return_5d")
    r5d_str = f"{r5d:.2%}" if r5d is not None else "N/A"
    sup = n1w.get("support_5d")
    res = n1w.get("resistance_5d")
    vol5 = n1w.get("volatility_5d")
    lines.append(
        f"1w signals: return={r5d_str}, trend={n1w.get('trend_5d')}, "
        f"support=${sup:.2f}, resistance=${res:.2f}, volatility={vol5:.4f}"
        if sup is not None and res is not None and vol5 is not None
        else f"1w signals: return={r5d_str}, trend={n1w.get('trend_5d')}"
    )

    tc = data["technical_context"]
    rsi = tc.get("rsi_14")
    rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
    bb_upper = tc.get("bollinger_upper")
    bb_lower = tc.get("bollinger_lower")
    bb_str = (
        f"${bb_lower:.2f}–${bb_upper:.2f} ({tc.get('bollinger_position')})"
        if bb_upper is not None and bb_lower is not None
        else tc.get("bollinger_position") or "N/A"
    )
    lines.append(
        f"Technicals: RSI(14)={rsi_str}, MACD={tc.get('macd_signal')}, "
        f"Bollinger={bb_str}"
    )

    vol20 = tc.get("volatility_20d")
    vol20_str = f"{vol20:.4f}" if vol20 is not None else "N/A"
    lines.append(f"20d volatility: {vol20_str}")

    return "\n".join(lines)
