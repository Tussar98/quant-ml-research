"""Tests for performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_ml.backtest.metrics import (
    cagr,
    calmar_ratio,
    compute_all,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)


class TestTotalReturn:
    def test_doubled_equity(self) -> None:
        eq = pd.Series([100, 150, 200])
        assert total_return(eq) == pytest.approx(1.0)

    def test_loss(self) -> None:
        eq = pd.Series([100, 80, 50])
        assert total_return(eq) == pytest.approx(-0.5)


class TestCAGR:
    def test_one_year_doubling(self) -> None:
        # 252 trading-day periods ⇒ 1 year ⇒ 100% CAGR
        eq = pd.Series(np.linspace(100, 200, 253))
        assert cagr(eq) == pytest.approx(1.0, rel=1e-3)

    def test_no_data_returns_zero(self) -> None:
        eq = pd.Series([100])
        assert cagr(eq) == 0.0


class TestSharpe:
    def test_constant_returns_zero_volatility(self) -> None:
        rets = pd.Series([0.001] * 100)
        # std == 0 -> Sharpe undefined, returns 0 by convention
        assert sharpe_ratio(rets) == 0.0

    def test_sharpe_sign(self) -> None:
        # Positive mean, mixed signs -> positive Sharpe
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.001, 0.01, 1000))
        assert sharpe_ratio(rets) > 0

    def test_sharpe_scaling(self) -> None:
        """Sharpe should be invariant to constant scaling of returns."""
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.001, 0.01, 1000))
        # Scaling returns by a constant scales mean and std equally, leaving
        # Sharpe unchanged.
        s1 = sharpe_ratio(rets)
        s2 = sharpe_ratio(rets * 2)
        assert s1 == pytest.approx(s2, rel=1e-10)


class TestSortino:
    def test_no_downside(self) -> None:
        rets = pd.Series([0.001] * 100)  # all positive
        assert sortino_ratio(rets) == 0.0


class TestMaxDrawdown:
    def test_monotonic_increase_no_drawdown(self) -> None:
        eq = pd.Series([100, 110, 120, 130])
        assert max_drawdown(eq) == 0.0

    def test_simple_drawdown(self) -> None:
        eq = pd.Series([100, 120, 60, 90])  # peak 120, trough 60 -> -50%
        assert max_drawdown(eq) == pytest.approx(-0.5)


class TestCalmar:
    def test_zero_drawdown_returns_zero(self) -> None:
        eq = pd.Series([100, 110, 120])
        assert calmar_ratio(eq) == 0.0


class TestWinRate:
    def test_half_winning(self) -> None:
        pnls = pd.Series([1, -1, 2, -2])
        assert win_rate(pnls) == 0.5

    def test_all_winners(self) -> None:
        pnls = pd.Series([1, 2, 3])
        assert win_rate(pnls) == 1.0

    def test_empty(self) -> None:
        assert win_rate(pd.Series(dtype=float)) == 0.0


class TestProfitFactor:
    def test_two_to_one(self) -> None:
        # Gains: 4, losses: 2 -> profit factor 2.0
        pnls = pd.Series([4, -2])
        assert profit_factor(pnls) == pytest.approx(2.0)

    def test_no_losses_infinite(self) -> None:
        pnls = pd.Series([1, 2, 3])
        assert profit_factor(pnls) == float("inf")


class TestExposure:
    def test_full_exposure(self) -> None:
        positions = pd.Series([1, 1, 1, 1])
        assert exposure(positions) == 1.0

    def test_partial(self) -> None:
        positions = pd.Series([0, 1, 1, 0])
        assert exposure(positions) == 0.5


class TestComputeAll:
    def test_returns_complete_metrics(self) -> None:
        eq = pd.Series(np.linspace(100, 110, 100))
        rets = eq.pct_change().fillna(0)
        positions = pd.Series([1] * 100)
        trade_pnls = pd.Series([10])
        m = compute_all(eq, rets, positions, trade_pnls)

        # Every field should be a finite number (not NaN)
        for k, v in m.to_dict().items():
            assert not (isinstance(v, float) and np.isnan(v)), f"{k} is NaN"
