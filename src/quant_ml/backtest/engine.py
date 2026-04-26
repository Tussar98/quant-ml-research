"""Backtest engine.

Executes a strategy's position series against historical prices with realistic
mechanics:
  - next-bar fills (no look-ahead)
  - commission + slippage costs
  - configurable position sizing
  - tracks equity curve, trades, exposure, drawdown

This is the contract that separates "strategy logic" from "execution logic":
the strategy decides WHAT to do; the engine simulates the consequences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_ml.backtest.costs import CostModel
from quant_ml.backtest.metrics import PerformanceMetrics, compute_all
from quant_ml.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series  # number of shares held
    trades: pd.DataFrame  # one row per executed trade
    metrics: PerformanceMetrics

    def summary(self) -> str:
        """Human-readable performance summary."""
        m = self.metrics
        lines = [
            "─" * 50,
            f"Total Return    : {m.total_return:>10.2%}",
            f"CAGR            : {m.cagr:>10.2%}",
            f"Volatility      : {m.volatility:>10.2%}",
            f"Sharpe          : {m.sharpe:>10.2f}",
            f"Sortino         : {m.sortino:>10.2f}",
            f"Max Drawdown    : {m.max_drawdown:>10.2%}",
            f"Calmar          : {m.calmar:>10.2f}",
            f"Win Rate        : {m.win_rate:>10.2%}",
            f"Profit Factor   : {m.profit_factor:>10.2f}",
            f"# Trades        : {m.n_trades:>10d}",
            f"Exposure        : {m.exposure:>10.2%}",
            "─" * 50,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Vectorized-with-loop hybrid backtest engine.

    The signal generation is vectorized. Trade execution loops bar-by-bar
    because position sizing, cash accounting, and cost application are state-
    dependent — vectorizing them introduces subtle bugs and is rarely worth
    the speedup at this scale.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        cost_model: CostModel | None = None,
        position_sizing: str = "fixed_fractional",
        fraction: float = 0.95,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if position_sizing not in {"fixed_fractional", "all_in"}:
            raise ValueError(f"unknown position_sizing: {position_sizing}")

        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.position_sizing = position_sizing
        self.fraction = fraction

    def _shares_to_buy(self, equity: float, price: float) -> int:
        """How many shares to purchase given current equity and price."""
        target_dollars = equity if self.position_sizing == "all_in" else equity * self.fraction
        return int(target_dollars // price)

    def run(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        execution_col: str = "Open",
        valuation_col: str = "Adj Close",
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        strategy : Strategy
            Strategy instance with `generate_signals(prices) -> pd.Series`.
        prices : pd.DataFrame
            OHLCV DataFrame, indexed by date.
        execution_col : str
            Column to use for trade fills. 'Open' is the standard — signals
            from bar t's close fire at bar t+1's open (the strategy already
            handles the t+1 lag, so we use bar t's open here in the loop).
        valuation_col : str
            Column to mark holdings to market each bar.
        """
        signals = strategy.generate_signals(prices)
        signals = signals.reindex(prices.index).fillna(0)

        # State variables
        cash = self.initial_capital
        shares = 0
        prev_target = 0  # the position state we're currently in

        equity = np.zeros(len(prices))
        position_record = np.zeros(len(prices))
        trade_records: list[dict[str, float | str | pd.Timestamp]] = []

        exec_prices = prices[execution_col].to_numpy()
        val_prices = prices[valuation_col].to_numpy()
        target_positions = signals.to_numpy()
        timestamps = prices.index

        for i in range(len(prices)):
            target = int(target_positions[i])
            exec_px = exec_prices[i]

            # Trade if our target position differs from current holdings
            # (long-only for now: target ∈ {0, 1})
            if target != prev_target and not np.isnan(exec_px):
                # Liquidate first if we hold something
                if shares > 0:
                    fill_px = self.cost_model.adjusted_fill_price(exec_px, side=-1)
                    proceeds = shares * fill_px
                    commission = self.cost_model.commission(shares * exec_px)
                    cash += proceeds - commission
                    trade_records.append(
                        {
                            "date": timestamps[i],
                            "side": "SELL",
                            "shares": shares,
                            "price": fill_px,
                            "commission": commission,
                            "cash_after": cash,
                        }
                    )
                    shares = 0

                # Enter new position if target is long
                if target == 1:
                    equity_now = cash  # no holdings at this instant
                    n = self._shares_to_buy(equity_now, exec_px)
                    if n > 0:
                        fill_px = self.cost_model.adjusted_fill_price(exec_px, side=1)
                        cost = n * fill_px
                        commission = self.cost_model.commission(n * exec_px)
                        if cost + commission <= cash:
                            cash -= cost + commission
                            shares = n
                            trade_records.append(
                                {
                                    "date": timestamps[i],
                                    "side": "BUY",
                                    "shares": n,
                                    "price": fill_px,
                                    "commission": commission,
                                    "cash_after": cash,
                                }
                            )

                prev_target = target

            # Mark to market
            holdings_value = shares * val_prices[i] if not np.isnan(val_prices[i]) else 0
            equity[i] = cash + holdings_value
            position_record[i] = shares

        equity_curve = pd.Series(equity, index=prices.index, name="equity")
        returns = equity_curve.pct_change().fillna(0)
        positions = pd.Series(position_record, index=prices.index, name="positions")
        trades = pd.DataFrame(trade_records)

        # Compute trade-level PnL by pairing buys and sells
        trade_pnls = self._compute_trade_pnls(trades)

        metrics = compute_all(
            equity=equity_curve,
            returns=returns,
            positions=positions,
            trade_pnls=trade_pnls,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
        )

    @staticmethod
    def _compute_trade_pnls(trades: pd.DataFrame) -> pd.Series:
        """Pair each SELL with the preceding BUY to compute round-trip PnLs."""
        if trades.empty:
            return pd.Series(dtype=float)

        pnls = []
        open_buy: dict[str, float] | None = None
        for _, row in trades.iterrows():
            if row["side"] == "BUY":
                open_buy = {"shares": row["shares"], "price": row["price"]}
            elif row["side"] == "SELL" and open_buy is not None:
                pnl = (row["price"] - open_buy["price"]) * open_buy["shares"]
                pnls.append(pnl)
                open_buy = None
        return pd.Series(pnls)
