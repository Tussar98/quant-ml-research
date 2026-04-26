"""Run walk-forward ML validation, then backtest the resulting signal stream.

This is the headline pipeline for the ML Engineer skill set:
  1. Build features and labels from raw prices
  2. Walk-forward train/predict (no look-ahead)
  3. Convert OOS predictions into a position series
  4. Backtest the position series with realistic costs
  5. Log everything to MLflow
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_ml.backtest import BacktestEngine, CostModel  # noqa: E402
from quant_ml.backtest.metrics import compute_all  # noqa: E402
from quant_ml.config import ExperimentConfig  # noqa: E402
from quant_ml.data import PriceLoader  # noqa: E402
from quant_ml.ml import (  # noqa: E402
    ExperimentTracker,
    WalkForwardValidator,
    build_dataset,
    make_classifier,
)
from quant_ml.strategies.base import Strategy  # noqa: E402


class _PrecomputedSignalStrategy(Strategy):
    """Adapter that returns precomputed walk-forward signals.

    Walk-forward produces a signal stream once, externally; this lets us reuse
    the BacktestEngine without retraining inside it.
    """

    name = "ml_walkforward"

    def __init__(self, signals: pd.Series) -> None:
        self._signals = signals

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return self._signals.reindex(prices.index).fillna(0)


def main(cfg: ExperimentConfig, enable_mlflow: bool = True) -> None:
    """Execute the walk-forward ML pipeline."""
    if cfg.walkforward is None:
        raise ValueError("Config has no `walkforward` section")

    # 1. Data
    loader = PriceLoader(cache_dir=cfg.data.cache_dir)
    ticker = cfg.data.tickers[0]
    prices = loader.load(ticker, cfg.data.start_date, cfg.data.end_date)
    logger.info(f"Loaded {len(prices)} bars for {ticker}")

    # 2. Build dataset
    X, y = build_dataset(prices, horizon=1)
    logger.info(f"Dataset: {X.shape}, label balance: {y.mean():.3f}")

    # 3. Walk-forward validation
    model = make_classifier(cfg.walkforward.model_type, cfg.walkforward.model_params)
    validator = WalkForwardValidator(
        n_splits=cfg.walkforward.n_splits,
        train_months=cfg.walkforward.train_months,
        test_months=cfg.walkforward.test_months,
        embargo_days=cfg.walkforward.embargo_days,
    )
    wf_result = validator.evaluate(model, X, y)
    logger.info(f"Walk-forward folds:\n{wf_result.fold_metrics}")
    logger.info(
        f"Mean OOS accuracy: {wf_result.mean_accuracy:.4f}  "
        f"(±{wf_result.std_accuracy:.4f})"
    )

    # 4. Convert OOS predictions into a long/flat position series
    # Lagged by 1 bar — predictions for day t are based on features as of day t-1
    # (already enforced inside walk-forward), but the strategy contract expects
    # an additional execution lag.
    threshold = 0.5
    long_signals = (wf_result.probabilities > threshold).astype(int)
    long_signals = long_signals.shift(1).fillna(0)

    # 5. Backtest the signal stream
    cost_model = CostModel(
        commission_pct=cfg.backtest.costs.commission_pct,
        slippage_bps=cfg.backtest.costs.slippage_bps,
    )
    engine = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        cost_model=cost_model,
        position_sizing=cfg.backtest.position_sizing,
        fraction=cfg.backtest.fraction,
    )
    strategy = _PrecomputedSignalStrategy(long_signals)
    bt_result = engine.run(strategy, prices)
    logger.info("\n" + bt_result.summary())

    # 6. Buy-and-hold benchmark
    bh_curve = (prices["Adj Close"] / prices["Adj Close"].iloc[0]) * cfg.backtest.initial_capital
    bh_returns = bh_curve.pct_change().fillna(0)
    bh_positions = pd.Series(1, index=prices.index)
    bh_metrics = compute_all(
        equity=bh_curve,
        returns=bh_returns,
        positions=bh_positions,
        trade_pnls=pd.Series([bh_curve.iloc[-1] - cfg.backtest.initial_capital]),
    )

    logger.info(f"\nBuy-and-Hold benchmark:\n{bh_metrics}")

    # 7. MLflow logging
    tracker = ExperimentTracker(
        experiment_name=cfg.name,
        tracking_uri=cfg.mlflow_tracking_uri,
        enabled=enable_mlflow,
    )
    with tracker.run(run_name=f"walkforward_{cfg.walkforward.model_type}_{ticker}"):
        tracker.log_params(
            {
                "ticker": ticker,
                "model_type": cfg.walkforward.model_type,
                "n_splits": cfg.walkforward.n_splits,
                "train_months": cfg.walkforward.train_months,
                "test_months": cfg.walkforward.test_months,
                "embargo_days": cfg.walkforward.embargo_days,
                **cfg.walkforward.model_params,
                "initial_capital": cfg.backtest.initial_capital,
                "commission_pct": cfg.backtest.costs.commission_pct,
                "slippage_bps": cfg.backtest.costs.slippage_bps,
            }
        )
        # Strategy metrics (prefixed)
        tracker.log_metrics(
            {f"strat_{k}": v for k, v in bt_result.metrics.to_dict().items() if isinstance(v, (int, float))}
        )
        # B&H benchmark metrics (prefixed)
        tracker.log_metrics(
            {f"bh_{k}": v for k, v in bh_metrics.to_dict().items() if isinstance(v, (int, float))}
        )
        # OOS classification quality
        tracker.log_metrics(
            {
                "oos_accuracy_mean": wf_result.mean_accuracy,
                "oos_accuracy_std": wf_result.std_accuracy,
            }
        )
        # Artifacts
        tracker.log_dataframe(wf_result.fold_metrics, "fold_metrics.csv")
        tracker.log_dataframe(bt_result.trades, "trades.csv")

        # Equity curve comparison plot
        fig, ax = plt.subplots(figsize=(10, 5))
        bt_result.equity_curve.plot(ax=ax, label="ML Strategy", color="C0")
        bh_curve.plot(ax=ax, label="Buy & Hold", color="gray", linestyle="--")
        ax.set_title(f"Walk-Forward {cfg.walkforward.model_type} on {ticker}")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        tracker.log_figure(fig, "equity_curve.png")
        plt.close(fig)

    # 8. Persist a local report
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    summary_path = reports_dir / f"{cfg.name}_summary.txt"
    summary_path.write_text(
        f"=== ML Strategy ({cfg.walkforward.model_type}) ===\n"
        f"{bt_result.summary()}\n\n"
        f"=== Buy & Hold ===\n"
        f"Total Return : {bh_metrics.total_return:.2%}\n"
        f"CAGR         : {bh_metrics.cagr:.2%}\n"
        f"Sharpe       : {bh_metrics.sharpe:.2f}\n"
        f"Max DD       : {bh_metrics.max_drawdown:.2%}\n\n"
        f"=== Walk-Forward Folds ===\n"
        f"{wf_result.fold_metrics.to_string()}\n"
        f"\nMean accuracy: {wf_result.mean_accuracy:.4f} (±{wf_result.std_accuracy:.4f})\n"
    )
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_walkforward.py <config.yaml>")
        sys.exit(1)
    cfg = ExperimentConfig.from_yaml(sys.argv[1])
    main(cfg)
