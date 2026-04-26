"""Tests for transaction cost model."""
from __future__ import annotations

import pytest

from quant_ml.backtest.costs import CostModel


class TestCostModel:
    def test_commission_calculation(self) -> None:
        cm = CostModel(commission_pct=0.001, slippage_bps=0)
        assert cm.commission(10_000) == pytest.approx(10.0)

    def test_slippage_calculation(self) -> None:
        cm = CostModel(commission_pct=0, slippage_bps=10)
        # 10 bps on $10,000 = $10
        assert cm.slippage(10_000) == pytest.approx(10.0)

    def test_slippage_pushes_buy_price_up(self) -> None:
        cm = CostModel(commission_pct=0, slippage_bps=10)
        adjusted = cm.adjusted_fill_price(100.0, side=1)
        assert adjusted > 100.0
        assert adjusted == pytest.approx(100.1)  # +10 bps

    def test_slippage_pushes_sell_price_down(self) -> None:
        cm = CostModel(commission_pct=0, slippage_bps=10)
        adjusted = cm.adjusted_fill_price(100.0, side=-1)
        assert adjusted < 100.0
        assert adjusted == pytest.approx(99.9)

    def test_negative_notional_handled(self) -> None:
        cm = CostModel(commission_pct=0.001, slippage_bps=10)
        # Costs should be on absolute notional
        assert cm.commission(-10_000) == pytest.approx(10.0)
        assert cm.slippage(-10_000) == pytest.approx(10.0)
