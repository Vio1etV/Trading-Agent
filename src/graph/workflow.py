"""Assemble the LangGraph workflow with intent-based routing.

Flow:
    START
      ↓
    router  (classifies the question)
      ↓
    conditional dispatch:
      - direct_reply already set (broker/vague/meta/out_of_scope) -> END
      - intent == "trade"     -> analyst -> risk_manager -> trader -> END
      - intent in {info,impact} -> analyst -> END
      - intent == "technical" -> risk_manager -> END
      - else                  -> END
"""

from __future__ import annotations

from langgraph.graph import START, END, StateGraph

from src.agents.analyst import load_analyst_model, make_analyst_node
from src.agents.risk_manager import make_risk_node
from src.agents.router import make_router_node
from src.agents.trader import load_trader_model, make_trader_node
from src.graph.state import TradingState


def build_workflow():
    """Load models, build nodes, wire the graph, return a compiled graph."""

    # Analyst, Risk Manager, and Router all use the same Qwen2.5-7B
    # weights — load once and share the model instance.
    print("Loading shared Qwen2.5-7B (router / analyst / risk)...")
    shared_model, shared_tokenizer = load_analyst_model()

    print("Loading trader model (Qwen3-8B)...")
    trader_model, trader_tokenizer = load_trader_model()

    router_node = make_router_node(shared_model, shared_tokenizer)
    analyst_node = make_analyst_node(shared_model, shared_tokenizer)
    risk_node = make_risk_node(shared_model, shared_tokenizer)
    trader_node = make_trader_node(trader_model, trader_tokenizer)

    graph = StateGraph(TradingState)
    graph.add_node("router", router_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("risk_manager", risk_node)
    graph.add_node("trader", trader_node)

    # --- Routing functions ---

    def route_after_router(state):
        # Router already wrote a direct reply (broker / meta / out_of_scope / vague) -> stop.
        if state.get("direct_reply"):
            return "end"

        intent = state.get("intent")
        if intent == "trade":
            return "trade"          # full pipeline
        if intent in ("impact", "info"):
            return "analyst_only"   # just one analyst answer
        if intent == "technical":
            return "risk_only"      # just one risk answer
        # Anything else (no direct_reply but unknown intent) -> stop with empty.
        return "end"

    def route_after_analyst(state):
        # Continue to risk only when this is a full trading pipeline.
        return "risk_manager" if state.get("intent") == "trade" else "end"

    def route_after_risk(state):
        return "trader" if state.get("intent") == "trade" else "end"

    # --- Wire the graph ---

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "end": END,
            "trade": "analyst",
            "analyst_only": "analyst",
            "risk_only": "risk_manager",
        },
    )
    graph.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "risk_manager": "risk_manager",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "risk_manager",
        route_after_risk,
        {
            "trader": "trader",
            "end": END,
        },
    )
    graph.add_edge("trader", END)

    return graph.compile()
