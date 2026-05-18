"""Assemble the LangGraph workflow from the agent nodes.

build_workflow() does everything: load models, build nodes, wire the
graph, and return a compiled graph you can .invoke().
"""

from __future__ import annotations

from langgraph.graph import START, END, StateGraph

from src.agents.analyst import load_analyst_model, make_analyst_node
from src.agents.risk_manager import load_risk_model, make_risk_node
from src.graph.state import TradingState


def build_workflow():
    """Load models, build nodes, wire the graph, return a compiled graph."""

    # 1. Load both models (expensive — happens once, at startup).
    print("Loading analyst model (Gemma-2-9B)...")
    analyst_model, analyst_tokenizer = load_analyst_model()

    print("Loading risk manager model (Qwen2.5-7B)...")
    risk_model, risk_tokenizer = load_risk_model()

    # 2. Build the node functions, binding each model in via the factory.
    analyst_node = make_analyst_node(analyst_model, analyst_tokenizer)
    risk_node = make_risk_node(risk_model, risk_tokenizer)

    # 3. Create the graph, telling LangGraph our state schema.
    graph = StateGraph(TradingState)

    # 4. Register the nodes. The string name is how edges refer to them.
    graph.add_node("analyst", analyst_node)
    graph.add_node("risk_manager", risk_node)

    # 5. Wire the edges. Sequential for now:
    #    START -> analyst -> risk_manager -> END
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "risk_manager")
    graph.add_edge("risk_manager", END)

    # 6. Compile into a runnable graph.
    return graph.compile()
