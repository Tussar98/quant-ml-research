"""Performance and risk metrics for backtest evaluation.

All functions take an equity curve or returns series and return scalars.
Annualization assumes 252 trading days/year (the US equity convention).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    """Standardized performance summary."""

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_trades: int
    exposure: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def total_return(equity: pd.Series) -> float:
    """Total return from start to end of the equity curve."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound Annual Growth Rate."""
    n_periods = len(equity) - 1
    if n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized standard deviation of returns."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio.

    risk_free_rate is the *annual* rate; we convert to per-period internally.
    """
    std = returns.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    # Guard against degenerate near-zero std on floating-point inputs
    if abs(excess.mean() / std) > 1e10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Sortino ratio: like Sharpe but only penalizes downside volatility."""
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction (e.g., -0.25)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Calmar = CAGR / |max drawdown|."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / abs(mdd))


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades with positive PnL."""
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).mean())


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss. >1 is profitable; >2 is strong."""
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = -trade_pnls[trade_pnls < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def exposure(positions: pd.Series) -> float:
    """Fraction of bars with a non-zero position. Lower = more selective."""
    if len(positions) == 0:
        return 0.0
    return float((positions != 0).mean())


def compute_all(
    equity: pd.Series,
    returns: pd.Series,
    positions: pd.Series,
    trade_pnls: pd.Series,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full performance summary in one call."""
    return PerformanceMetrics(
        total_return=total_return(equity),
        cagr=cagr(equity),
        volatility=annualized_volatility(returns),
        sharpe=sharpe_ratio(returns, risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate),
        max_drawdown=max_drawdown(equity),
        calmar=calmar_ratio(equity),
        win_rate=win_rate(trade_pnls),
        profit_factor=profit_factor(trade_pnls),
        n_trades=len(trade_pnls),
        exposure=exposure(positions),
    )
