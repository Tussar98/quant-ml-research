"""Strategy base class — every strategy implements `generate_signals`.

A strategy maps a price DataFrame to a position series:
    +1 = long, 0 = flat, -1 = short

The backtest engine executes those positions with proper next-bar fills
and cost accounting. Separating strategy logic from execution is critical:
it lets us swap strategies without touching the backtest engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate target positions from price data.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV DataFrame indexed by date.

        Returns
        -------
        pd.Series
            Target position at each bar: +1 long, 0 flat, -1 short.
            MUST be lagged appropriately to avoid look-ahead bias.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
