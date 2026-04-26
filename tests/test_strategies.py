"""Tests for trading strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_ml.strategies import MACrossover, RSIMeanReversion


class TestMACrossover:
    def test_invalid_windows(self) -> None:
        with pytest.raises(ValueError):
            MACrossover(fast=50, slow=10)
        with pytest.raises(ValueError):
            MACrossover(fast=10, slow=10)

    def test_signals_are_lagged(self, synthetic_prices: pd.DataFrame) -> None:
        """First signal should be 0 (no lookahead from start)."""
        strat = MACrossover(fast=5, slow=20)
        signals = strat.generate_signals(synthetic_prices)
        assert signals.iloc[0] == 0

    def test_signals_are_binary(self, synthetic_prices: pd.DataFrame) -> None:
        strat = MACrossover()
        signals = strat.generate_signals(synthetic_prices)
        assert set(signals.unique()).issubset({0, 1})

    def test_index_alignment(self, synthetic_prices: pd.DataFrame) -> None:
        strat = MACrossover()
        signals = strat.generate_signals(synthetic_prices)
        assert signals.index.equals(synthetic_prices.index)

    def test_uptrend_eventually_long(self, trending_prices: pd.DataFrame) -> None:
        """In a clean uptrend, the strategy should be long most of the time."""
        strat = MACrossover(fast=10, slow=50)
        signals = strat.generate_signals(trending_prices)
        # After warmup, most signals should be long
        post_warmup = signals.iloc[60:]
        assert post_warmup.mean() > 0.8


class TestRSIMeanReversion:
    def test_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=80, overbought=20)
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=-1, overbought=70)

    def test_signals_are_binary(self, synthetic_prices: pd.DataFrame) -> None:
        strat = RSIMeanReversion()
        signals = strat.generate_signals(synthetic_prices)
        assert set(signals.unique()).issubset({0.0, 1.0})

    def test_lagged_signals(self, synthetic_prices: pd.DataFrame) -> None:
        strat = RSIMeanReversion()
        signals = strat.generate_signals(synthetic_prices)
        assert signals.iloc[0] == 0

    def test_long_only(self, synthetic_prices: pd.DataFrame) -> None:
        """Mean reversion strategy is long-only by construction."""
        strat = RSIMeanReversion()
        signals = strat.generate_signals(synthetic_prices)
        assert (signals >= 0).all()

    def test_state_machine_holds_position(self) -> None:
        """Once long, the strategy stays long until an overbought signal."""
        # Construct prices with a sharp consecutive-down phase to drive RSI <30,
        # followed by a flat plateau (RSI stays in middle, never hits 70).
        n = 100
        idx = pd.bdate_range("2020-01-01", periods=n)
        # 30 strict declines followed by 70 flat days
        decline = np.linspace(100, 30, 30)
        flat = np.full(70, 30.0) + np.linspace(0, 0.5, 70)  # tiny drift, no surge
        close = pd.Series(np.concatenate([decline, flat]), index=idx)
        prices = pd.DataFrame({"Adj Close": close})
        strat = RSIMeanReversion(window=14, oversold=30, overbought=70)
        signals = strat.generate_signals(prices)
        # During the decline RSI hits oversold; during flat phase it stays mid.
        # Strategy enters long during decline and never gets an overbought exit,
        # so it should still be long at the end.
        assert signals.iloc[-1] == 1
