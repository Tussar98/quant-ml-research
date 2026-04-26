# Extensions

These are sketches for next-weekend work. Not implemented in v1, but the package's APIs are designed to plug into them cleanly.

## Streamlit Dashboard (planned)

```python
# dashboard/app.py — sketch
import streamlit as st
from quant_ml.data import PriceLoader
from quant_ml.strategies import MACrossover, RSIMeanReversion
from quant_ml.backtest import BacktestEngine, CostModel

st.title("Quant ML Backtest Explorer")

ticker = st.sidebar.selectbox("Ticker", ["AAPL", "MSFT", "GOOGL"])
strategy_name = st.sidebar.selectbox("Strategy", ["MA Crossover", "RSI Mean Reversion"])
fast = st.sidebar.slider("Fast MA", 5, 30, 10)
slow = st.sidebar.slider("Slow MA", 20, 200, 50)

# ... wire up to BacktestEngine, plot equity curve, render metrics table
```

To implement:
1. Run `pip install -e ".[deploy]"`
2. Build `dashboard/app.py` calling the existing public API
3. Deploy free on Streamlit Cloud or Hugging Face Spaces

## FastAPI Service (planned)

```python
# api/main.py — sketch
from fastapi import FastAPI
from quant_ml.config import ExperimentConfig
from scripts.run_backtest import main as run_backtest

app = FastAPI()

@app.post("/backtest")
def backtest(config: ExperimentConfig):
    result = run_backtest(config)
    return result.metrics.to_dict()
```

Containerize with Docker, deploy to Fly.io free tier.

## Multi-asset portfolio backtests (planned)

The current `BacktestEngine` is single-asset. Extending to multi-asset requires:
1. `Strategy.generate_signals` returning a DataFrame (one column per asset)
2. Engine maintains a position vector instead of scalar
3. Cash accounting unchanged; cost model per-asset

Should be a 1-2 day refactor — the math layers out cleanly.
