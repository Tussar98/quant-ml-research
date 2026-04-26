"""End-to-end smoke test with synthetic data.

We don't have network access here, so we synthesize AAPL-like prices,
write them to the cache directory, then run both pipelines (backtest +
walk-forward) so we can capture realistic-looking output for the REPORT.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

ROOT = Path("/home/claude/quant-ml-research")
sys.path.insert(0, str(ROOT / "src"))

from quant_ml.backtest import BacktestEngine, CostModel
from quant_ml.backtest.metrics import compute_all
from quant_ml.data.loader import PriceLoader
from quant_ml.ml import (
    WalkForwardValidator,
    build_dataset,
    make_classifier,
)
from quant_ml.strategies import MACrossover, RSIMeanReversion
from quant_ml.strategies.base import Strategy


def make_synthetic_aapl_like(seed: int = 7) -> pd.DataFrame:
    """Synthetic 'AAPL-like' OHLCV: 13 years of daily bars with trend + vol regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", "2022-12-30")
    n = len(idx)

    # Mix two regimes: low-vol bull and a couple of vol spikes
    base_drift = 0.0006
    base_vol = 0.014
    rets = rng.normal(base_drift, base_vol, n)

    # Inject a 2020-COVID-style vol spike
    covid_start = idx.get_loc(pd.Timestamp("2020-02-20"))
    covid_end = idx.get_loc(pd.Timestamp("2020-04-30"))
    rets[covid_start:covid_end] = rng.normal(-0.001, 0.05, covid_end - covid_start)

    # Inject a 2022 drawdown
    bear_start = idx.get_loc(pd.Timestamp("2022-01-04"))
    bear_end = idx.get_loc(pd.Timestamp("2022-09-30"))
    rets[bear_start:bear_end] = rng.normal(-0.0005, 0.02, bear_end - bear_start)

    close = 30.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(50_000_000, 200_000_000, n),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def seed_cache(loader: PriceLoader, ticker: str, start: date, end: date) -> pd.DataFrame:
    """Seed the parquet cache with synthetic data."""
    df = make_synthetic_aapl_like()
    cache_path = loader._cache_path(ticker, start, end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


class _PrecomputedSignalStrategy(Strategy):
    name = "ml_walkforward"

    def __init__(self, signals: pd.Series) -> None:
        self._signals = signals

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return self._signals.reindex(prices.index).fillna(0)


def run_full_pipeline() -> dict:
    """Run all three strategies + B&H on the same data, return numbers."""
    cache_dir = ROOT / "data" / "cache"
    loader = PriceLoader(cache_dir=cache_dir)

    start = date(2010, 1, 1)
    end = date(2022, 12, 31)
    ticker = "AAPL"

    prices = seed_cache(loader, ticker, start, end)
    print(f"Synthetic prices: {len(prices)} bars from {prices.index[0].date()} to {prices.index[-1].date()}")

    cost_model = CostModel(commission_pct=0.0005, slippage_bps=2.0)
    engine = BacktestEngine(
        initial_capital=100_000,
        cost_model=cost_model,
        position_sizing="fixed_fractional",
        fraction=0.95,
    )

    results = {}

    # 1. MA Crossover
    ma_strat = MACrossover(fast=10, slow=50)
    ma_result = engine.run(ma_strat, prices)
    results["ma_crossover"] = ma_result
    print("\n=== MA Crossover ===")
    print(ma_result.summary())

    # 2. RSI Mean Reversion
    rsi_strat = RSIMeanReversion(window=14, oversold=30, overbought=70)
    rsi_result = engine.run(rsi_strat, prices)
    results["rsi_mean_reversion"] = rsi_result
    print("\n=== RSI Mean Reversion ===")
    print(rsi_result.summary())

    # 3. Walk-forward ML
    print("\n=== Walk-Forward ML (Random Forest) ===")
    X, y = build_dataset(prices, horizon=1)
    model = make_classifier("random_forest", {"n_estimators": 100, "max_depth": 5})
    validator = WalkForwardValidator(
        n_splits=5, train_months=36, test_months=6, embargo_days=5
    )
    wf = validator.evaluate(model, X, y)
    print(wf.fold_metrics.to_string())
    print(f"\nMean OOS accuracy: {wf.mean_accuracy:.4f} (±{wf.std_accuracy:.4f})")

    # Convert OOS probs to long/flat positions
    long_signals = (wf.probabilities > 0.5).astype(int).shift(1).fillna(0)
    ml_strat = _PrecomputedSignalStrategy(long_signals)
    ml_result = engine.run(ml_strat, prices)
    results["ml_walkforward"] = ml_result
    results["wf_folds"] = wf.fold_metrics
    results["wf_mean_acc"] = wf.mean_accuracy
    results["wf_std_acc"] = wf.std_accuracy
    print(ml_result.summary())

    # 4. Buy-and-hold baseline
    bh_curve = (prices["Adj Close"] / prices["Adj Close"].iloc[0]) * 100_000
    bh_returns = bh_curve.pct_change().fillna(0)
    bh_positions = pd.Series(1, index=prices.index)
    bh_metrics = compute_all(
        equity=bh_curve,
        returns=bh_returns,
        positions=bh_positions,
        trade_pnls=pd.Series([bh_curve.iloc[-1] - 100_000]),
    )
    results["buy_and_hold"] = bh_metrics
    print("\n=== Buy & Hold ===")
    print(f"Total Return    : {bh_metrics.total_return:.2%}")
    print(f"CAGR            : {bh_metrics.cagr:.2%}")
    print(f"Sharpe          : {bh_metrics.sharpe:.2f}")
    print(f"Max Drawdown    : {bh_metrics.max_drawdown:.2%}")

    return results


if __name__ == "__main__":
    results = run_full_pipeline()

    # Save a comparison plot to reports/
    import matplotlib.pyplot as plt

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Equity curves
    fig, ax = plt.subplots(figsize=(11, 6))
    results["ma_crossover"].equity_curve.plot(ax=ax, label="MA Crossover", color="C0")
    results["rsi_mean_reversion"].equity_curve.plot(
        ax=ax, label="RSI Mean Reversion", color="C1"
    )
    results["ml_walkforward"].equity_curve.plot(
        ax=ax, label="ML Walk-Forward", color="C2"
    )
    # B&H
    prices_path = list((ROOT / "data" / "cache").glob("*.parquet"))[0]
    prices = pd.read_parquet(prices_path)
    bh_curve = (prices["Adj Close"] / prices["Adj Close"].iloc[0]) * 100_000
    bh_curve.plot(ax=ax, label="Buy & Hold", color="gray", linestyle="--")
    ax.set_title("Strategy Comparison: $100k starting capital, 5bps commission + 2bps slippage")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "equity_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved equity curves to {figures_dir / 'equity_curves.png'}")

    # Save numbers as JSON for the REPORT
    import json

    summary = {}
    for name in ["ma_crossover", "rsi_mean_reversion", "ml_walkforward"]:
        summary[name] = results[name].metrics.to_dict()
    summary["buy_and_hold"] = results["buy_and_hold"].to_dict()
    summary["wf_mean_acc"] = float(results["wf_mean_acc"])
    summary["wf_std_acc"] = float(results["wf_std_acc"])
    summary["wf_folds"] = results["wf_folds"].to_dict(orient="records")

    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"Saved summary to {reports_dir / 'summary.json'}")
