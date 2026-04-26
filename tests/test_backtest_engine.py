"""Tests for the backtest engine.

The engine is the most error-prone component (cash accounting, look-ahead,
position state). Test it on cases with known answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_ml.backtest import BacktestEngine, CostModel
from quant_ml.strategies.base import Strategy


class _ConstantSignalStrategy(Strategy):
    """Strategy that returns a fixed precomputed signal series — for testing."""

    def __init__(self, signal: pd.Series) -> None:
        self._signal = signal

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return self._signal.reindex(prices.index).fillna(0)


class TestBacktestEngineBasics:
    def test_initial_capital_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BacktestEngine(initial_capital=0)

    def test_unknown_position_sizing(self) -> None:
        with pytest.raises(ValueError, match="position_sizing"):
            BacktestEngine(position_sizing="kelly_unsupported")

    def test_no_signals_means_all_cash(self, synthetic_prices: pd.DataFrame) -> None:
        """If we never enter, equity should equal initial capital throughout."""
        signal = pd.Series(0, index=synthetic_prices.index)
        strategy = _ConstantSignalStrategy(signal)
        engine = BacktestEngine(initial_capital=10_000, cost_model=CostModel(0, 0))
        result = engine.run(strategy, synthetic_prices)

        assert result.equity_curve.iloc[0] == pytest.approx(10_000)
        assert result.equity_curve.iloc[-1] == pytest.approx(10_000)
        assert result.metrics.n_trades == 0


class TestBacktestEngineKnownPnL:
    """Construct cases where we know the right answer analytically."""

    def test_buy_and_hold_with_no_costs(self) -> None:
        """Buy on day 1, hold to end. PnL = (exit - entry) * shares."""
        idx = pd.bdate_range("2020-01-01", periods=10)
        # Constant prices except for a clean rise on day 5
        close = pd.Series([100, 100, 100, 100, 100, 110, 110, 110, 110, 110], index=idx)
        prices = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Adj Close": close,
                "Volume": 1_000_000,
            }
        )
        # Always-long signal (entered on day 0)
        signal = pd.Series(1, index=idx)
        strategy = _ConstantSignalStrategy(signal)

        engine = BacktestEngine(
            initial_capital=10_000,
            cost_model=CostModel(commission_pct=0, slippage_bps=0),
            position_sizing="fixed_fractional",
            fraction=1.0,
        )
        result = engine.run(strategy, prices)

        # Day 0: buy 100 shares @ $100 = $10,000 invested, $0 cash
        # Day 5: holdings = 100 * $110 = $11,000
        # Final equity: $11,000
        assert len(result.trades) == 1  # one BUY (no SELL — never exited)
        assert result.equity_curve.iloc[-1] == pytest.approx(11_000, abs=1)

    def test_round_trip_pnl_with_no_costs(self) -> None:
        """Buy at $100, sell at $110 => +10% on capital deployed."""
        idx = pd.bdate_range("2020-01-01", periods=10)
        close = pd.Series([100, 100, 100, 100, 100, 110, 110, 110, 110, 110], index=idx)
        prices = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Adj Close": close,
                "Volume": 1_000_000,
            }
        )
        # Long days 1-5, flat thereafter
        signal = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=idx)
        strategy = _ConstantSignalStrategy(signal)

        engine = BacktestEngine(
            initial_capital=10_000,
            cost_model=CostModel(0, 0),
            position_sizing="fixed_fractional",
            fraction=1.0,
        )
        result = engine.run(strategy, prices)

        # Buy 100 sh @ $100 (day 0), sell 100 sh @ $110 (day 5)
        # Final cash should be $11,000
        assert result.equity_curve.iloc[-1] == pytest.approx(11_000, abs=1)
        assert len(result.trades) == 2  # BUY row + SELL row
        assert result.metrics.n_trades == 1  # one round-trip pair

    def test_costs_reduce_returns(self) -> None:
        """With costs, the same trade should produce less profit."""
        idx = pd.bdate_range("2020-01-01", periods=10)
        close = pd.Series([100, 100, 100, 100, 100, 110, 110, 110, 110, 110], index=idx)
        prices = pd.DataFrame(
            {
                "Open": close, "High": close, "Low": close,
                "Close": close, "Adj Close": close, "Volume": 1_000_000,
            }
        )
        signal = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=idx)
        strategy = _ConstantSignalStrategy(signal)

        no_cost = BacktestEngine(initial_capital=10_000, cost_model=CostModel(0, 0), fraction=1.0)
        with_cost = BacktestEngine(
            initial_capital=10_000,
            cost_model=CostModel(commission_pct=0.001, slippage_bps=10),
            fraction=1.0,
        )
        r_free = no_cost.run(strategy, prices)
        r_paid = with_cost.run(strategy, prices)

        assert r_paid.equity_curve.iloc[-1] < r_free.equity_curve.iloc[-1]


class TestBacktestEngineNoLookAhead:
    """The engine must not act on signals from the same bar's close."""

    def test_signal_lag_one_bar(self) -> None:
        """A signal emitted on bar t is filled at bar t (using bar t's open).
        The strategy itself is responsible for shifting; verify the engine
        doesn't peek at future bars.
        """
        idx = pd.bdate_range("2020-01-01", periods=5)
        # Open prices distinguishable from close prices
        opens = pd.Series([100, 102, 104, 106, 108], index=idx)
        closes = pd.Series([101, 103, 105, 107, 109], index=idx)
        prices = pd.DataFrame(
            {"Open": opens, "High": closes, "Low": opens,
             "Close": closes, "Adj Close": closes, "Volume": 1_000_000}
        )
        # Signal: long from day 1 onwards (not day 0)
        signal = pd.Series([0, 1, 1, 1, 1], index=idx)
        strategy = _ConstantSignalStrategy(signal)

        engine = BacktestEngine(
            initial_capital=10_000,
            cost_model=CostModel(0, 0),
            fraction=1.0,
        )
        result = engine.run(strategy, prices)

        # First trade should fill at day 1's OPEN ($102), not day 0's
        first_buy = result.trades.iloc[0]
        assert first_buy["side"] == "BUY"
        assert first_buy["price"] == pytest.approx(102.0)


class TestBacktestResult:
    def test_summary_string_contains_key_metrics(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        signal = pd.Series(1, index=synthetic_prices.index)
        strategy = _ConstantSignalStrategy(signal)
        engine = BacktestEngine(initial_capital=10_000, cost_model=CostModel(0, 0))
        result = engine.run(strategy, synthetic_prices)

        summary = result.summary()
        for key in ["Sharpe", "Max Drawdown", "Total Return", "Calmar"]:
            assert key in summary

    def test_returns_align_with_equity(self, synthetic_prices: pd.DataFrame) -> None:
        signal = pd.Series(1, index=synthetic_prices.index)
        strategy = _ConstantSignalStrategy(signal)
        engine = BacktestEngine(initial_capital=10_000, cost_model=CostModel(0, 0))
        result = engine.run(strategy, synthetic_prices)

        # First-bar return should be 0 (no prior bar)
        assert result.returns.iloc[0] == 0
        # Returns reconstructed from equity should match the stored series
        recomputed = result.equity_curve.pct_change().fillna(0)
        np.testing.assert_allclose(result.returns.values, recomputed.values, atol=1e-10)
