"""Transaction cost models.

Real backtests must include costs. A strategy that's profitable on paper but
unprofitable after 5bps of slippage isn't actually profitable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Linear transaction cost model.

    Attributes
    ----------
    commission_pct : float
        Commission as a fraction of trade notional (e.g., 0.0005 = 5 bps).
    slippage_bps : float
        Slippage in basis points applied to each fill. Models the price
        impact of crossing the spread plus market impact.
    """

    commission_pct: float = 0.0005
    slippage_bps: float = 2.0

    def commission(self, notional: float) -> float:
        """Dollar commission on a trade of `notional` dollars (absolute)."""
        return abs(notional) * self.commission_pct

    def slippage(self, notional: float) -> float:
        """Dollar slippage cost on a trade."""
        return abs(notional) * self.slippage_bps * 1e-4

    def total_cost(self, notional: float) -> float:
        """Total cost (commission + slippage) for a trade."""
        return self.commission(notional) + self.slippage(notional)

    def adjusted_fill_price(self, mid_price: float, side: int) -> float:
        """Apply slippage to the fill price.

        Parameters
        ----------
        mid_price : float
            Midpoint price (e.g., the next bar's open).
        side : int
            +1 for buy, -1 for sell.
        """
        slip = self.slippage_bps * 1e-4
        return mid_price * (1 + side * slip)
