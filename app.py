"""Streamlit chat frontend for the Trading Agent.

Run with:
    streamlit run app.py --server.port 6006 --server.address 0.0.0.0
"""

from __future__ import annotations

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
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def escape_dollars(text: str) -> str:
    """Stop Streamlit's markdown from treating $...$ as LaTeX math."""
    return text.replace("$", "\\$") if text else text


def get_cached_tickers() -> list[str]:
    cache_dir = PROJECT_ROOT / "data" / "cache"
    if not cache_dir.exists():
        return []
    return sorted(p.stem.upper() for p in cache_dir.glob("*.json"))


def derive_final_response(result: dict) -> tuple[str, str]:
    """Pick the right field from the graph result and a short label."""
    if result.get("direct_reply"):
        return result["direct_reply"], "router"
    if result.get("trader_recommendation"):
        return result["trader_recommendation"], "trader"
    if result.get("analyst_report"):
        return result["analyst_report"], "analyst"
    if result.get("risk_report"):
        return result["risk_report"], "risk"
    return (
        "Sorry, I couldn't generate a response. Try rephrasing.",
        "fallback",
    )


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.graph.workflow import build_workflow
    return build_workflow()


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
- **Trade decisions** — "Should I buy NVDA?" → full analyst + risk + trader report
- **Event impact** — "How does the Iran situation affect NVDA?"
- **Technicals** — "What's the technical setup for TSLA?"
- **Company info** — "What does UNH do?"
- **Beginner Qs** — "Where do I buy stocks?"

Off-topic questions (games, weather, etc.) I'll politely decline.
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
        if msg.get("source"):
            st.caption(f"_{msg['source']}_")
        st.markdown(escape_dollars(msg["content"]))

# Input pinned at the bottom.
user_input = st.chat_input("Ask about a stock... e.g. 'Should I buy NVDA?'")

if user_input:
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the graph.
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

        placeholder.empty()
        st.caption(f"_intent: {intent} · source: {source} · {elapsed:.1f}s_")
        st.markdown(escape_dollars(response))

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "source": f"intent: {intent} · {source}",
        })
