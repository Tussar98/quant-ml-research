"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Deterministic synthetic OHLCV series."""
    np.random.seed(42)
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = np.random.normal(0.0005, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Adj Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def trending_prices() -> pd.DataFrame:
    """A clean uptrend — useful for sanity-checking long strategies."""
    n = 252
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * (1.001 ** np.arange(n))  # ~28% annual
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Adj Close": close,
            "Volume": np.full(n, 1_000_000),
        },
        index=idx,
    )
    return df
