"""Technical indicators as pure functions.

Each function takes a price series (or DataFrame) and returns a Series.
No state, no side effects, easy to test, easy to compose into pipelines.

These are deliberately implemented from scratch — `pandas-ta` is great but
having the math visible signals you understand what RSI/MACD actually compute,
not just that you know which library to import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def returns(close: pd.Series, periods: int = 1, log: bool = False) -> pd.Series:
    """Simple or log returns over `periods` bars."""
    if log:
        return np.log(close / close.shift(periods))
    return close.pct_change(periods=periods)


def realized_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling realized volatility from log returns.

    Annualized assuming 252 trading days when annualize=True.
    """
    log_r = np.log(close / close.shift(1))
    vol = log_r.rolling(window=window, min_periods=window).std()
    return vol * np.sqrt(252) if annualize else vol


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Wilder's smoothing differs from a simple rolling mean — it's an EMA with
    alpha = 1/window. This matches the standard charting-platform RSI.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (alpha = 1/window)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper, lower, and %B position."""
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    # %B: position within the band, useful as an ML feature
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct": pct_b})


def build_feature_matrix(prices: pd.DataFrame, close_col: str = "Adj Close") -> pd.DataFrame:
    """Compose all features into a single DataFrame ready for ML.

    This is the canonical feature set used across strategies and ML models.
    Keeps feature engineering centralized so train/test/prod see identical
    transformations — a real production concern.
    """
    close = prices[close_col]
    features = pd.DataFrame(index=prices.index)

    # Returns at multiple horizons
    features["ret_1d"] = returns(close, 1)
    features["ret_5d"] = returns(close, 5)
    features["ret_20d"] = returns(close, 20)

    # Trend
    features["sma_10"] = sma(close, 10)
    features["sma_50"] = sma(close, 50)
    features["sma_ratio"] = features["sma_10"] / features["sma_50"] - 1

    # Momentum
    features["rsi_14"] = rsi(close, 14)

    # Volatility
    features["vol_20d"] = realized_volatility(close, 20)

    # MACD
    features = features.join(macd(close))

    # Bollinger %B (only the position feature; bands themselves are price-scale)
    bb = bollinger_bands(close)
    features["bb_pct"] = bb["bb_pct"]

    return features
