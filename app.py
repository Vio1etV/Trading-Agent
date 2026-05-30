"""Streamlit chat frontend for the Trading Agent.

Run with:
    streamlit run app.py --server.port 6006 --server.address 0.0.0.0
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ActiveTrader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS for the trade-output cards ───────────────────────────────────────────
st.markdown("""
<style>
.action-buy, .action-sell, .action-hold {
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 0.5rem 1.4rem;
    border-radius: 10px;
    display: inline-block;
    letter-spacing: 0.05em;
}
.action-buy  { background: linear-gradient(135deg, #059669, #10b981); }
.action-sell { background: linear-gradient(135deg, #dc2626, #ef4444); }
.action-hold { background: linear-gradient(135deg, #d97706, #f59e0b); }

.metric-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 120px;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.55;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.0rem;
    font-weight: 600;
}

.case-col {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.case-bull { border-left: 4px solid #10b981; }
.case-bear { border-left: 4px solid #ef4444; }
.case-title { font-weight: 700; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def escape_dollars(text: str) -> str:
    return text.replace("$", "\\$") if text else text


def get_cached_tickers() -> list[str]:
    cache_dir = PROJECT_ROOT / "data" / "cache"
    if not cache_dir.exists():
        return []
    return sorted(p.stem.upper() for p in cache_dir.glob("*.json"))


def load_price_history(ticker: str) -> list[dict]:
    path = PROJECT_ROOT / "data" / "cache" / f"{ticker.upper()}.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("price_history", []) or []
    except Exception:
        return []


def parse_field(text: str, field: str) -> str:
    """Pull a single Field: value line out of the trader's text."""
    if not text:
        return "N/A"
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return "N/A"


def parse_action(text: str) -> str:
    raw = parse_field(text, "Action").lower()
    if "buy" in raw:
        return "buy"
    if "sell" in raw:
        return "sell"
    return "hold"


def parse_section(text: str, header: str) -> str:
    """Extract everything after `Header:` until the next blank line or known header."""
    if not text:
        return ""
    lines = text.split("\n")
    out = []
    capturing = False
    known_next = ("Action:", "Confidence:", "Entry Zone:", "Stop Loss:",
                  "Target Price:", "Reasoning:", "Key Risks:")
    for line in lines:
        stripped = line.strip()
        if not capturing and stripped.startswith(f"{header}:"):
            capturing = True
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                out.append(rest)
            continue
        if capturing:
            if any(stripped.startswith(h) for h in known_next):
                break
            out.append(line)
    return "\n".join(out).strip()


def derive_final_response(result: dict) -> tuple[str, str]:
    if result.get("direct_reply"):
        return result["direct_reply"], "router"
    if result.get("trader_recommendation"):
        return result["trader_recommendation"], "trader"
    if result.get("analyst_report"):
        return result["analyst_report"], "analyst"
    if result.get("risk_report"):
        return result["risk_report"], "risk"
    return "Sorry, I couldn't generate a response. Try rephrasing.", "fallback"


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.graph.workflow import build_workflow
    return build_workflow()


# ── Renderers for the trader output ──────────────────────────────────────────

def _action_badge(action: str) -> str:
    return f'<span class="action-{action}">{action.upper()}</span>'


def _metric_row(text: str) -> str:
    fields = [
        ("Confidence", parse_field(text, "Confidence")),
        ("Entry Zone", parse_field(text, "Entry Zone")),
        ("Stop Loss", parse_field(text, "Stop Loss")),
        ("Target Price", parse_field(text, "Target Price")),
    ]
    cards = "".join(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
        for label, value in fields
    )
    return f'<div class="metric-row">{cards}</div>'


def render_format_a(text: str) -> None:
    """Format A: big action badge + metric cards row + reasoning + risks."""
    action = parse_action(text)
    st.markdown(_action_badge(action), unsafe_allow_html=True)
    st.markdown(escape_dollars(_metric_row(text)), unsafe_allow_html=True)

    reasoning = parse_section(text, "Reasoning")
    risks = parse_section(text, "Key Risks")
    if reasoning:
        st.markdown("**Reasoning**")
        st.markdown(escape_dollars(reasoning))
    if risks:
        st.markdown("**Key Risks**")
        st.markdown(escape_dollars(risks))


def render_format_b(text: str) -> None:
    """Format B: action badge + side-by-side bull/bear columns + metrics row."""
    action = parse_action(text)
    st.markdown(_action_badge(action), unsafe_allow_html=True)

    reasoning = parse_section(text, "Reasoning")
    risks = parse_section(text, "Key Risks")

    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.markdown(
            '<div class="case-col case-bull">'
            '<div class="case-title">📈 Bull case</div>'
            + escape_dollars(reasoning or "_(none)_")
            + '</div>',
            unsafe_allow_html=True,
        )
    with col_bear:
        st.markdown(
            '<div class="case-col case-bear">'
            '<div class="case-title">📉 Bear case</div>'
            + escape_dollars(risks or "_(none)_")
            + '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(escape_dollars(_metric_row(text)), unsafe_allow_html=True)


def render_price_chart(ticker: str) -> None:
    """Render the last 14 days of close prices + 5 day linear projection."""
    history = load_price_history(ticker)
    if not history:
        st.caption(
            f"_No cached price history for {ticker}. "
            "Run `python scripts/prepare_data.py --ticker " + ticker + "` to enable the chart._"
        )
        return

    try:
        import numpy as np
        import pandas as pd
        import altair as alt
    except ImportError:
        st.caption("_Charting libraries not installed (pandas / altair)._")
        return

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(14).reset_index(drop=True)

    # Simple linear extrapolation from the last 5 days.
    proj_rows = []
    if len(df) >= 3:
        n = min(5, len(df))
        recent = df.tail(n)
        x = np.arange(n)
        y = recent["close"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        last_date = df["date"].iloc[-1]
        for i in range(1, 6):
            proj_rows.append({
                "date": last_date + pd.Timedelta(days=i),
                "close": float(slope * (n - 1 + i) + intercept),
            })

    df["series"] = "actual"
    proj_df = pd.DataFrame(proj_rows)
    if not proj_df.empty:
        proj_df["series"] = "projected"
    combined = pd.concat([df, proj_df], ignore_index=True)

    base = alt.Chart(combined).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("close:Q", title="Close ($)"),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["actual", "projected"],
            range=["#3b82f6", "#9ca3af"],
        ), legend=alt.Legend(title=None)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(
            domain=["actual", "projected"],
            range=[[1, 0], [4, 4]],
        ), legend=None),
    )
    chart = (base.mark_line(point=True)).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    st.caption("_Dashed line = simple linear extrapolation, not a real forecast._")


def render_trade_output(text: str, ticker: str, fmt: str) -> None:
    """Render the trader output in either format A or B, plus the price chart."""
    if fmt == "B":
        render_format_b(text)
    else:
        render_format_a(text)
    st.markdown("---")
    st.markdown(f"**14-day price + 5-day projection · {ticker}**")
    render_price_chart(ticker)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Default ticker")
    cached_tickers = get_cached_tickers()
    if cached_tickers:
        default_ticker = st.selectbox(
            "Used when your question doesn't mention one",
            cached_tickers,
            index=cached_tickers.index("NVDA") if "NVDA" in cached_tickers else 0,
        )
    else:
        default_ticker = st.text_input("Default ticker", value="NVDA").upper()

    st.markdown("---")
    st.markdown("### 🤖 What I can do")
    st.markdown(
        """
- **Trade decisions** — "Should I buy NVDA?"
- **Event impact** — "How does X affect NVDA?"
- **Technicals** — "Technical setup for TSLA?"
- **Company info** — "What does UNH do?"
- **Beginner Qs** — "Where do I buy stocks?"

Off-topic questions I'll politely decline.
        """
    )

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p style="opacity:0.4; font-size:0.75rem;">'
        'CS 496 · Agent AI · Northwestern University</p>',
        unsafe_allow_html=True,
    )


# ── Main: chat ───────────────────────────────────────────────────────────────

st.title("📊 ActiveTrader")
st.caption("Multi-agent stock analysis — ask me anything about stocks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("caption"):
            st.caption(msg["caption"])
        if msg.get("kind") == "trade":
            render_trade_output(msg["content"], msg["ticker"], msg["fmt"])
        else:
            st.markdown(escape_dollars(msg["content"]))

user_input = st.chat_input("Ask about a stock... e.g. 'Should I buy NVDA?'")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Routing your question..._")

        with st.spinner("Thinking..."):
            graph = load_pipeline()
            start = time.time()
            result = graph.invoke({
                "user_question": user_input,
                "ticker": default_ticker,
                "intent": None,
                "direct_reply": None,
                "analyst_report": None,
                "risk_report": None,
                "trader_recommendation": None,
                "final_response": None,
            })
            elapsed = time.time() - start

        response, source = derive_final_response(result)
        intent = result.get("intent") or "unknown"
        ticker = result.get("ticker") or default_ticker
        caption = f"_intent: {intent} · source: {source} · {elapsed:.1f}s_"

        placeholder.empty()
        st.caption(caption)

        if intent == "trade" and source == "trader":
            fmt = random.choice(["A", "B"])
            render_trade_output(response, ticker, fmt)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "caption": caption,
                "kind": "trade",
                "ticker": ticker,
                "fmt": fmt,
            })
        else:
            st.markdown(escape_dollars(response))
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "caption": caption,
            })
