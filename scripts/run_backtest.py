"""Run a single backtest from a config.

Loads data, instantiates the strategy, runs the backtest engine, logs to
MLflow, and writes a summary report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

# Make the package importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_ml.backtest import BacktestEngine, CostModel  # noqa: E402
from quant_ml.config import ExperimentConfig  # noqa: E402
from quant_ml.data import PriceLoader  # noqa: E402
from quant_ml.ml.tracking import ExperimentTracker  # noqa: E402
from quant_ml.strategies import MACrossover, RSIMeanReversion  # noqa: E402


STRATEGY_REGISTRY = {
    "ma_crossover": MACrossover,
    "rsi_mean_reversion": RSIMeanReversion,
}


def main(cfg: ExperimentConfig, enable_mlflow: bool = True) -> None:
    """Execute backtest workflow."""
    if cfg.strategy.name == "ml_classifier":
        logger.error("For ml_classifier strategy, use the walkforward command instead")
        sys.exit(1)

    # 1. Data
    loader = PriceLoader(cache_dir=cfg.data.cache_dir)
    ticker = cfg.data.tickers[0]
    prices = loader.load(ticker, cfg.data.start_date, cfg.data.end_date)

    # 2. Strategy
    strategy_cls = STRATEGY_REGISTRY[cfg.strategy.name]
    strategy = strategy_cls(**cfg.strategy.params)
    logger.info(f"Strategy: {strategy}")

    # 3. Backtest engine
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

    result = engine.run(strategy, prices)
    logger.info("\n" + result.summary())

    # 4. MLflow logging
    tracker = ExperimentTracker(
        experiment_name=cfg.name,
        tracking_uri=cfg.mlflow_tracking_uri,
        enabled=enable_mlflow,
    )
    with tracker.run(run_name=f"{cfg.strategy.name}_{ticker}"):
        tracker.log_params(
            {
                "ticker": ticker,
                "strategy": cfg.strategy.name,
                **cfg.strategy.params,
                "initial_capital": cfg.backtest.initial_capital,
                "commission_pct": cfg.backtest.costs.commission_pct,
                "slippage_bps": cfg.backtest.costs.slippage_bps,
                "start_date": str(cfg.data.start_date),
                "end_date": str(cfg.data.end_date),
            }
        )
        tracker.log_metrics(result.metrics.to_dict())

        # Equity curve figure
        fig, ax = plt.subplots(figsize=(10, 5))
        result.equity_curve.plot(ax=ax, label="Strategy", color="C0")
        # Buy-and-hold benchmark
        bh = (prices["Adj Close"] / prices["Adj Close"].iloc[0]) * cfg.backtest.initial_capital
        bh.plot(ax=ax, label="Buy & Hold", color="gray", linestyle="--")
        ax.set_title(f"{cfg.strategy.name} on {ticker}")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        tracker.log_figure(fig, "equity_curve.png")
        plt.close(fig)

        tracker.log_dataframe(result.trades, "trades.csv")

    # 5. Persist report locally too
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    summary_path = reports_dir / f"{cfg.name}_summary.txt"
    summary_path.write_text(result.summary())
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_backtest.py <config.yaml>")
        sys.exit(1)
    cfg = ExperimentConfig.from_yaml(sys.argv[1])
    main(cfg)
