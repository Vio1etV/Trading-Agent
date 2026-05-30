"""Streamlit web frontend for the Trading Agent.

Run with:
    streamlit run app.py

Requires all models downloaded and cached data available.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Add project root to path
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

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global */
.stApp {
    font-family: 'Outfit', sans-serif;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}
.hero h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.hero p {
    font-size: 1rem;
    opacity: 0.6;
    margin-top: 0;
}

/* Agent cards */
.agent-card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}
.agent-card:hover {
    border-color: rgba(128,128,128,0.4);
}
.agent-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}
.agent-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-analyst {
    background: rgba(59,130,246,0.15);
    color: #3b82f6;
}
.badge-risk {
    background: rgba(245,158,11,0.15);
    color: #f59e0b;
}
.badge-trader {
    background: rgba(16,185,129,0.15);
    color: #10b981;
}

/* Action badges */
.action-buy {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 8px;
    display: inline-block;
}
.action-sell {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 8px;
    display: inline-block;
}
.action-hold {
    background: linear-gradient(135deg, #d97706, #f59e0b);
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 8px;
    display: inline-block;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.5;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.4rem;
}
.status-running { background: #f59e0b; animation: pulse 1.5s infinite; }
.status-done { background: #10b981; }
.status-waiting { background: rgba(128,128,128,0.3); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    opacity: 0.4;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────

def parse_action(recommendation: str) -> str:
    """Extract Buy/Hold/Sell from the recommendation text."""
    for line in recommendation.split("\n"):
        if line.strip().startswith("Action:"):
            action = line.split(":", 1)[1].strip().lower()
            if "buy" in action:
                return "buy"
            elif "sell" in action:
                return "sell"
            else:
                return "hold"
    return "hold"


def parse_field(text: str, field: str) -> str:
    """Extract a field value from a report."""
    for line in text.split("\n"):
        if line.strip().startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return "N/A"


def escape_dollars(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't treat them as LaTeX delimiters."""
    return text.replace("$", "\\$") if text else text


def get_cached_tickers() -> list[str]:
    """Return list of tickers with cached data."""
    cache_dir = PROJECT_ROOT / "data" / "cache"
    if not cache_dir.exists():
        return []
    return sorted(p.stem.upper() for p in cache_dir.glob("*.json"))


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load all models and build the workflow. Cached across reruns."""
    from src.graph.workflow import build_workflow
    return build_workflow()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    cached_tickers = get_cached_tickers()

    if cached_tickers:
        ticker = st.selectbox(
            "Select ticker",
            cached_tickers,
            index=cached_tickers.index("NVDA") if "NVDA" in cached_tickers else 0,
        )
    else:
        ticker = st.text_input("Ticker symbol", value="NVDA").upper()

    question = st.text_input(
        "Your question",
        value=f"Should I buy {ticker} right now?",
    )

    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Agent Architecture")
    st.markdown("""
    **Analyst** → Qwen2.5-7B  
    Fundamentals, news, sentiment, macro

    **Risk Manager** → Qwen2.5-7B  
    Technical indicators, support/resistance

    **Trader** → Qwen3-8B *(SFT fine-tuned)*  
    Synthesizes both → Buy / Hold / Sell
    """)

    st.markdown("---")
    st.markdown(
        '<p style="opacity:0.4; font-size:0.75rem;">'
        'CS 496 · Agent AI · Northwestern University</p>',
        unsafe_allow_html=True,
    )


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    '<h1>📊 ActiveTrader</h1>'
    '<p>Multi-agent stock analysis powered by LLMs</p>'
    '</div>',
    unsafe_allow_html=True,
)

if run_button:
    # ── Load models ──────────────────────────────────────────────────────
    with st.status("Loading models...", expanded=True) as status:
        st.write("Loading Analyst, Risk Manager, and Trader models...")
        graph = load_pipeline()
        status.update(label="Models loaded ✓", state="complete")

    # ── Run pipeline ─────────────────────────────────────────────────────
    col_analyst, col_risk = st.columns(2)

    # Analyst
    with col_analyst:
        st.markdown(
            '<div class="agent-card">'
            '<div class="agent-header">'
            '<span class="agent-badge badge-analyst">Analyst</span>'
            '<span style="opacity:0.5; font-size:0.8rem;">Qwen2.5-7B</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        analyst_placeholder = st.empty()
        analyst_placeholder.info("⏳ Running analyst...")

    # Risk Manager
    with col_risk:
        st.markdown(
            '<div class="agent-card">'
            '<div class="agent-header">'
            '<span class="agent-badge badge-risk">Risk Manager</span>'
            '<span style="opacity:0.5; font-size:0.8rem;">Qwen2.5-7B</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        risk_placeholder = st.empty()
        risk_placeholder.info("⏳ Running risk manager...")

    # Run the full graph
    start_time = time.time()
    result = graph.invoke(
        {
            "ticker": ticker,
            "user_question": question,
            "analyst_report": None,
            "risk_report": None,
            "trader_recommendation": None,
        }
    )
    elapsed = time.time() - start_time

    # Update analyst card
    with col_analyst:
        analyst_placeholder.empty()
        st.markdown(escape_dollars(result.get("analyst_report", "No report generated.")))

    # Update risk card
    with col_risk:
        risk_placeholder.empty()
        st.markdown(escape_dollars(result.get("risk_report", "No report generated.")))

    # ── Trader Recommendation ────────────────────────────────────────────
    st.markdown("---")

    recommendation = result.get("trader_recommendation", "")
    action = parse_action(recommendation)

    # Action badge
    action_class = f"action-{action}"
    action_label = action.upper()

    st.markdown(
        '<div class="agent-card" style="border-color: rgba(16,185,129,0.3);">'
        '<div class="agent-header">'
        '<span class="agent-badge badge-trader">Trader</span>'
        '<span style="opacity:0.5; font-size:0.8rem;">Qwen3-8B · SFT fine-tuned</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Metrics row
    entry_zone = parse_field(recommendation, "Entry Zone")
    stop_loss = parse_field(recommendation, "Stop Loss")
    target_price = parse_field(recommendation, "Target Price")
    confidence = parse_field(recommendation, "Confidence")

    st.markdown(
        f'<div class="metric-row">'
        f'<div class="metric-card">'
        f'<div class="metric-label">Action</div>'
        f'<div class="{action_class}">{action_label}</div>'
        f'</div>'
        f'<div class="metric-card">'
        f'<div class="metric-label">Confidence</div>'
        f'<div class="metric-value">{confidence}</div>'
        f'</div>'
        f'<div class="metric-card">'
        f'<div class="metric-label">Entry Zone</div>'
        f'<div class="metric-value">{entry_zone}</div>'
        f'</div>'
        f'<div class="metric-card">'
        f'<div class="metric-label">Stop Loss</div>'
        f'<div class="metric-value">{stop_loss}</div>'
        f'</div>'
        f'<div class="metric-card">'
        f'<div class="metric-label">Target Price</div>'
        f'<div class="metric-value">{target_price}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Full recommendation
    st.markdown(escape_dollars(recommendation))

    # Footer
    st.markdown(
        f'<div class="footer">'
        f'Analysis completed in {elapsed:.1f}s · 3 agents · {ticker}'
        f'</div>',
        unsafe_allow_html=True,
    )

else:
    # Landing state
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="agent-card">'
            '<div class="agent-header">'
            '<span class="agent-badge badge-analyst">Analyst</span>'
            '</div>'
            '<p style="opacity:0.6; font-size:0.9rem;">'
            'Reads fundamentals, news, social sentiment, and macro context '
            'to produce a comprehensive analyst report.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="agent-card">'
            '<div class="agent-header">'
            '<span class="agent-badge badge-risk">Risk Manager</span>'
            '</div>'
            '<p style="opacity:0.6; font-size:0.9rem;">'
            'Analyzes technical indicators — RSI, MACD, Bollinger Bands — '
            'to assess risk and identify support/resistance.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="agent-card">'
            '<div class="agent-header">'
            '<span class="agent-badge badge-trader">Trader</span>'
            '</div>'
            '<p style="opacity:0.6; font-size:0.9rem;">'
            'SFT fine-tuned on GPT-4o recommendations. Synthesizes both '
            'reports into an actionable Buy / Hold / Sell decision.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">'
        'Select a ticker and click <b>Run Analysis</b> to begin.'
        '</div>',
        unsafe_allow_html=True,
    )
