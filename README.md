# Trading Agent

## What we're building

A user asks a question about a stock (e.g. "Should I buy NVDA right now?"),
and the system automatically pulls data, runs analysis, and returns a
**structured trading recommendation**: Buy / Hold / Sell, entry zone, stop
loss, target price, reasoning, and risks. The end goal is a web app where
users ask questions, watch the agents analyze step by step, and get a
professional-style report.

This is a two-week class project — the focus is a **working demo**, not a
production product.

## The idea: 3 agents

```
User question
   │
   ├──→ Analyst Agent      (Gemma-2-9B)    fundamentals / news / social / macro → analyst report
   ├──→ Risk Manager Agent (Qwen2.5-7B)    technical indicators                 → risk report
   │
   └──→ Trader Agent       (Qwen3-8B, SFT fine-tuned)
            takes both reports + the user question → final recommendation
```

- The first two agents are **not trained** — just an open model + a prompt.
- The Trader is the one we **SFT fine-tune** (the training focus of the project).
- **LangGraph** wires the three agents together.

## How to run

```bash
git clone <repo> && cd Trading-Agent
# activate your conda env
export HF_TOKEN=hf_xxx
bash setup.sh                                   # install deps + download 3 models (~49GB)
python scripts/prepare_data.py --ticker NVDA    # fetch data, save to cache (needs internet)
python scripts/test_agents.py --ticker NVDA     # run the agents, see the reports (needs GPU)
```

## What's in the repo

| Path | Purpose |
|------|---------|
| `src/tools/` | 6 data tools: price, technical indicators, fundamentals, finance news, social sentiment, macro news |
| `src/tools/data_cache.py` | Reads cached data (yfinance gets rate-limited on cloud servers, so we split it: fetch first, then use) |
| `src/agents/analyst.py` | Analyst agent — writes the analysis report with Gemma |
| `src/agents/risk_manager.py` | Risk Manager agent — writes the risk report with Qwen2.5 |
| `src/graph/state.py` | The shared data structure for LangGraph |
| `src/graph/workflow.py` | Wires the agent nodes into a flow graph |
| `scripts/prepare_data.py` | Fetch data and save it to cache (run on a machine with internet) |
| `scripts/test_agents.py` | End-to-end test of the two agents |

## Status

- [x] 6 data tools
- [x] Data cache layer
- [x] Analyst + Risk Manager agents (running on GPU)
- [x] LangGraph wiring the two agents together
- [ ] Trader agent
- [ ] Generate SFT training data
- [ ] SFT fine-tune the Trader
- [ ] Streamlit web frontend

## End goal

A working web demo: type a stock question → watch three agents collaborate
→ get a trading recommendation with price levels and reasoning. The Trader
is a model we SFT-trained ourselves — that's the core thing the project
shows off.
