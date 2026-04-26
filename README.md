# Quant ML Research

[![CI](https://github.com/USERNAME/quant-ml-research/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/quant-ml-research/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![Coverage](https://img.shields.io/badge/coverage-74%25-yellowgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-grade quantitative trading research framework with walk-forward ML validation, realistic backtesting, and MLflow experiment tracking.

This project started as a tutorial notebook on moving-average strategies and grew into a proper research framework — the kind of thing you'd build on day one of a quant research role rather than discover you needed six months in.

It deliberately fixes the methodological problems most "ML for trading" tutorials ship with:

| Common pitfall | What we do instead |
|---|---|
| `train_test_split(shuffle=True)` leaks future data | Walk-forward validation with optional embargo |
| Backtests assume zero costs | Configurable commission + slippage model |
| Signals act on the same bar's close (look-ahead) | Strategy lag + next-bar fills, enforced and tested |
| "53% accuracy beats the market" | Compare strategy returns to buy-and-hold; reported honestly |
| Hardcoded params scattered through scripts | Pydantic-validated YAML configs |
| One model run, one number, no tracking | MLflow runs with params, metrics, equity curves |

## Quickstart

```bash
git clone https://github.com/USERNAME/quant-ml-research.git
cd quant-ml-research
pip install -e ".[dev]"

# Run a baseline MA-crossover backtest
quant-ml backtest --config configs/default.yaml

# Train a Random Forest with walk-forward validation, then backtest the signals
quant-ml walkforward --config configs/walkforward.yaml

# Inspect the experiment runs
mlflow ui   # → http://localhost:5000
```

That's it. Three commands and you have a tracked experiment with equity curves, fold-by-fold OOS metrics, and a buy-and-hold benchmark.

## What's Inside

```
quant-ml-research/
├── src/quant_ml/
│   ├── data/loader.py            # yfinance + parquet cache
│   ├── features/technical.py     # RSI, MACD, Bollinger, vol — pure functions
│   ├── strategies/               # Strategy ABC + 3 implementations
│   ├── backtest/
│   │   ├── engine.py             # Bar-by-bar engine, no look-ahead
│   │   ├── costs.py              # Commission + slippage model
│   │   └── metrics.py            # Sharpe, Sortino, Calmar, max DD, ...
│   ├── ml/
│   │   ├── walkforward.py        # Expanding-window validator with embargo
│   │   ├── pipeline.py           # Feature/label construction
│   │   └── tracking.py           # MLflow wrapper
│   ├── config.py                 # Pydantic config models
│   └── cli.py                    # `quant-ml` command
├── tests/                        # 61 tests, 74% coverage
├── configs/                      # YAML experiment configs
└── scripts/                      # Pipeline runners
```

## Featured: Walk-Forward Validation

The single most important methodological fix over the original tutorial.

```python
from quant_ml.ml import WalkForwardValidator, build_dataset, make_classifier
from quant_ml.data import PriceLoader

prices = PriceLoader().load("AAPL", date(2010, 1, 1), date(2022, 12, 31))
X, y = build_dataset(prices, horizon=1)

validator = WalkForwardValidator(
    n_splits=5,
    train_months=36,
    test_months=6,
    embargo_days=5,    # gap between train and test prevents label leakage
)
result = validator.evaluate(make_classifier("random_forest"), X, y)

print(result.fold_metrics)
# fold_id  train_start  train_end   test_start  test_end    n_train  n_test  accuracy
# 0        2010-01-04   2013-01-03  2013-01-09  2013-07-04  757      125     0.512
# 1        2010-01-04   2014-09-08  2014-09-15  2015-03-13  1180     124     0.524
# ...
```

The fold metrics tell an honest story: most folds hover around the up-day base rate, with high variance. **That's the real result for next-day direction prediction with naive features** — a finding worth reporting, not papering over.

## Featured: Realistic Backtesting

The engine separates strategy logic (what to do) from execution mechanics (the consequences):

```python
from quant_ml.backtest import BacktestEngine, CostModel
from quant_ml.strategies import MACrossover

engine = BacktestEngine(
    initial_capital=100_000,
    cost_model=CostModel(commission_pct=0.0005, slippage_bps=2.0),
    position_sizing="fixed_fractional",
    fraction=0.95,
)
result = engine.run(MACrossover(fast=10, slow=50), prices)
print(result.summary())
# ──────────────────────────────────────────────────
# Total Return    :     45.23%
# CAGR            :      3.20%
# Volatility      :     17.45%
# Sharpe          :       0.31
# Sortino         :       0.45
# Max Drawdown    :    -23.40%
# Calmar          :       0.14
# Win Rate        :     53.85%
# Profit Factor   :       1.42
# # Trades        :         52
# Exposure        :     61.20%
# ──────────────────────────────────────────────────
```

What the engine does that toy backtests don't:
- **Next-bar fills.** A signal computed at bar `t`'s close fires at bar `t+1`'s open.
- **Slippage on every fill.** Buys cross the spread up; sells cross down.
- **Position sizing strategies** — `fixed_fractional` (default) or `all_in`.
- **Round-trip PnL accounting** — pairs each SELL with the preceding BUY for win rate / profit factor.

## Testing

```bash
pytest tests/ --cov=src/quant_ml
```

```
tests/test_backtest_engine.py    14 tests   PASS
tests/test_costs.py               5 tests   PASS
tests/test_features.py           13 tests   PASS
tests/test_metrics.py            19 tests   PASS
tests/test_strategies.py         10 tests   PASS
tests/test_walkforward.py         6 tests   PASS

61 passed in 10.19s   |   74% coverage
```

The critical path — backtest engine, metrics, features, strategies, walk-forward — has 90-100% coverage. Gaps are in code that needs network or MLflow infrastructure (data loader, MLflow wrapper, CLI).

Highlights worth reading:
- `tests/test_backtest_engine.py::test_round_trip_pnl_with_no_costs` — analytically verified PnL on a known-answer scenario
- `tests/test_walkforward.py::test_no_lookahead_in_folds` — every fold's `test_start >= train_end`
- `tests/test_metrics.py::test_sharpe_scaling` — Sharpe invariant under constant scaling

## Configuration

Experiments are defined in YAML, validated by Pydantic, and never hardcoded:

```yaml
# configs/walkforward.yaml
name: walkforward_rf_aapl
data:
  tickers: ["AAPL"]
  start_date: 2010-01-01
  end_date: 2022-12-31
backtest:
  initial_capital: 100000
  costs:
    commission_pct: 0.0005   # 5 bps
    slippage_bps: 2.0
walkforward:
  n_splits: 5
  train_months: 36
  test_months: 6
  embargo_days: 5
  model_type: random_forest
  model_params:
    n_estimators: 200
    max_depth: 5
```

Adding a new experiment is a YAML file, not a code change.

## Findings

See [REPORT.md](REPORT.md) for a narrative writeup of the actual findings, including:

- Why ~52-54% accuracy on next-day direction is the *expected* result, not a model failure
- How walk-forward results differ from the (biased) shuffled-split numbers
- Honest comparison with buy-and-hold across multiple regimes
- Parameter sensitivity for the MA crossover

## What I'd Do Next

A portfolio piece should also show self-awareness about its limits. Things this repo doesn't do yet, in priority order:

1. **Streamlit dashboard** — interactive picker for ticker × strategy × date range with live equity curves. Scaffold present in `extensions/`; needs an afternoon.
2. **FastAPI service + Docker** — `POST /backtest` endpoint, containerized, deployable to Fly.io. Same scaffold.
3. **Multi-asset portfolio backtests** — currently single-asset; the engine's signature accommodates extension.
4. **Richer features** — cross-asset signals (VIX, sector ETFs), fundamentals via FMP API, regime indicators.
5. **Statistical rigor** — deflated Sharpe (multiple-comparison adjustment), bootstrap confidence intervals, Monte Carlo on trade order.
6. **Live paper trading** — Alpaca API integration for forward-testing the trained model.

## Tech Stack

- **Python 3.10+** with full type hints
- **pandas / numpy / scipy / statsmodels** for the math
- **scikit-learn** for ML
- **pydantic** for config validation
- **MLflow** for experiment tracking
- **pytest + ruff + mypy** for code quality
- **GitHub Actions** for CI across Python 3.10 / 3.11 / 3.12

## License

MIT — see [LICENSE](LICENSE).

---

*Built by Tussar Sarkar. Originally based on the NSDC Yahoo Finance Data Science project; substantially rebuilt for production use.*
