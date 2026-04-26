"""RSI Mean Reversion Strategy (long-only).

Enter long when RSI < oversold threshold, exit when RSI > exit threshold.
Standard 30/70 thresholds by default.
"""
from __future__ import annotations

import pandas as pd

from quant_ml.features.technical import rsi
from quant_ml.strategies.base import Strategy


class RSIMeanReversion(Strategy):
    """Long-only RSI mean reversion strategy."""

    name = "rsi_mean_reversion"

    def __init__(
        self,
        window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        close_col: str = "Adj Close",
    ) -> None:
        if not 0 < oversold < overbought < 100:
            raise ValueError("Require 0 < oversold < overbought < 100")
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.close_col = close_col

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices[self.close_col]
        rsi_values = rsi(close, self.window)

        # Build raw signals: 1 = enter long, -1 = exit (treat as flat for long-only)
        raw = pd.Series(0, index=close.index, dtype=float)
        raw.loc[rsi_values < self.oversold] = 1.0
        raw.loc[rsi_values > self.overbought] = -1.0

        # Long-only: convert exit signals (-1) into "flat" (0) and forward-fill
        # so we hold once long until an exit signal flips us flat.
        positions = raw.replace(-1.0, 0.0)
        # Carry forward last non-zero state... but raw.replace also turns
        # explicit exits into 0, so we need a state machine:
        state = 0
        out = []
        for r in raw.values:
            if r == 1:
                state = 1
            elif r == -1:
                state = 0
            out.append(state)
        positions = pd.Series(out, index=close.index, dtype=float)

        # Lag by 1 bar to avoid look-ahead
        return positions.shift(1).fillna(0)
