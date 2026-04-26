"""Moving Average Crossover Strategy.

Long when fast MA > slow MA, flat otherwise. The original notebook strategy
done right: signals are computed from prior-bar data and lagged one bar so
trades execute at the next bar's open (no look-ahead).
"""
from __future__ import annotations

import pandas as pd

from quant_ml.features.technical import sma
from quant_ml.strategies.base import Strategy


class MACrossover(Strategy):
    """Long-only moving-average crossover."""

    name = "ma_crossover"

    def __init__(self, fast: int = 10, slow: int = 50, close_col: str = "Adj Close") -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.close_col = close_col

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices[self.close_col]
        fast_ma = sma(close, self.fast)
        slow_ma = sma(close, self.slow)

        # Raw signal: long when fast > slow, flat otherwise
        signal = (fast_ma > slow_ma).astype(int)

        # Lag by 1 bar — signal computed using bar t's close is acted on at t+1's open
        return signal.shift(1).fillna(0)
