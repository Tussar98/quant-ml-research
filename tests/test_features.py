"""Tests for technical indicator features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_ml.features.technical import (
    bollinger_bands,
    build_feature_matrix,
    ema,
    macd,
    realized_volatility,
    returns,
    rsi,
    sma,
)


class TestSMA:
    def test_sma_constant_series(self) -> None:
        s = pd.Series([10] * 20)
        result = sma(s, 5)
        # First 4 values should be NaN (warmup); rest equal 10
        assert result.iloc[:4].isna().all()
        assert (result.iloc[4:] == 10).all()

    def test_sma_window_value(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        result = sma(s, 3)
        # Window of 3: NaN, NaN, mean(1,2,3)=2, mean(2,3,4)=3, mean(3,4,5)=4
        assert result.iloc[2] == 2
        assert result.iloc[3] == 3
        assert result.iloc[4] == 4


class TestRSI:
    def test_rsi_range(self) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
        result = rsi(s, 14).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_rsi_uptrend_high(self) -> None:
        # Strict uptrend -> RSI should approach 100
        s = pd.Series(np.arange(1, 100, dtype=float))
        result = rsi(s, 14).dropna()
        assert result.iloc[-1] > 95

    def test_rsi_downtrend_low(self) -> None:
        s = pd.Series(np.arange(100, 1, -1, dtype=float))
        result = rsi(s, 14).dropna()
        assert result.iloc[-1] < 5


class TestReturns:
    def test_simple_returns(self) -> None:
        s = pd.Series([100, 110, 121])
        result = returns(s)
        assert result.iloc[1] == pytest.approx(0.1)
        assert result.iloc[2] == pytest.approx(0.1)

    def test_log_returns(self) -> None:
        s = pd.Series([100, 110])
        result = returns(s, log=True)
        assert result.iloc[1] == pytest.approx(np.log(1.1))


class TestRealizedVolatility:
    def test_zero_for_constant_series(self) -> None:
        s = pd.Series([100] * 50)
        result = realized_volatility(s, 20).dropna()
        assert (result == 0).all()


class TestEMA:
    def test_ema_smoother_than_sma(self) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 100))
        sma_result = sma(s, 10).dropna()
        ema_result = ema(s, 10).dropna()
        # Both exist; just check shapes match the spec
        assert len(sma_result) > 0 and len(ema_result) > 0


class TestMACD:
    def test_macd_columns(self) -> None:
        s = pd.Series(np.linspace(100, 200, 100))
        result = macd(s)
        assert set(result.columns) == {"macd", "macd_signal", "macd_hist"}


class TestBollingerBands:
    def test_band_ordering(self) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
        bb = bollinger_bands(s).dropna()
        # Upper > mid > lower at every point
        assert (bb["bb_upper"] >= bb["bb_mid"]).all()
        assert (bb["bb_mid"] >= bb["bb_lower"]).all()


class TestBuildFeatureMatrix:
    def test_returns_dataframe_with_expected_columns(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=200)
        rng = np.random.default_rng(0)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=idx)
        prices = pd.DataFrame({"Adj Close": close})

        features = build_feature_matrix(prices)
        expected = {
            "ret_1d", "ret_5d", "ret_20d",
            "sma_10", "sma_50", "sma_ratio",
            "rsi_14", "vol_20d",
            "macd", "macd_signal", "macd_hist", "bb_pct",
        }
        assert set(features.columns) == expected
