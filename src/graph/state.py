"""Shared state for the trading agent graph.

State is the dict that flows through every node. LangGraph needs a
schema (a TypedDict here) so it knows what fields exist. Each node
reads the fields it needs and returns the fields it wants to update.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class TradingState(TypedDict):
    # --- Inputs: set once at the start, never change ---
    ticker: str
    user_question: str

    # --- Outputs: start as None, filled in by each agent node ---
    analyst_report: Optional[str]        # filled by analyst_node
    risk_report: Optional[str]           # filled by risk_manager_node
    trader_recommendation: Optional[str]  # filled by trader_node (later)
