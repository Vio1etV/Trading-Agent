"""Shared state for the trading agent graph.

State is the dict that flows through every node. LangGraph needs a
schema (a TypedDict here) so it knows what fields exist. Each node
reads the fields it needs and returns the fields it wants to update.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class TradingState(TypedDict):
    # --- Input ---
    user_question: str

    # --- Filled by the router ---
    ticker: Optional[str]            # extracted from question, or fallback to UI selection
    intent: Optional[str]            # trade | impact | technical | info | broker | vague | meta | out_of_scope
    direct_reply: Optional[str]      # short answer the router writes directly (no agents needed)

    # --- Filled by the agents (only when needed) ---
    analyst_report: Optional[str]
    risk_report: Optional[str]
    trader_recommendation: Optional[str]

    # --- Final user-facing response ---
    final_response: Optional[str]
