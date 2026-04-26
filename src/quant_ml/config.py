"""Typed configuration models loaded from YAML.

Pydantic gives us validation, defaults, and IDE autocomplete for configs.
This is how production codebases handle environment separation — no more
hardcoded magic numbers scattered through the code.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """How to fetch and cache price data."""

    tickers: list[str] = Field(..., min_length=1)
    start_date: date
    end_date: date
    cache_dir: Path = Path("data/cache")
    use_adjusted: bool = True

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info: object) -> date:  # noqa: ARG003
        # Cross-field validation handled at model level below
        return v

    def model_post_init(self, __context: object) -> None:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")


class CostConfig(BaseModel):
    """Transaction cost model.

    commission_pct: fraction of trade notional (e.g., 0.0005 = 5 bps)
    slippage_bps: market-impact estimate in basis points
    """

    commission_pct: float = Field(0.0005, ge=0, le=0.05)
    slippage_bps: float = Field(2.0, ge=0)


class BacktestConfig(BaseModel):
    """Backtest engine parameters."""

    initial_capital: float = Field(100_000, gt=0)
    position_sizing: Literal["fixed_fractional", "all_in"] = "fixed_fractional"
    fraction: float = Field(0.95, gt=0, le=1.0)
    costs: CostConfig = CostConfig()


class StrategyConfig(BaseModel):
    """Which strategy to backtest and its hyperparameters."""

    name: Literal["ma_crossover", "rsi_mean_reversion", "ml_classifier"]
    params: dict[str, float | int | str] = Field(default_factory=dict)


class WalkForwardConfig(BaseModel):
    """Walk-forward ML validation parameters."""

    n_splits: int = Field(5, ge=2, le=20)
    train_months: int = Field(36, ge=6)
    test_months: int = Field(6, ge=1)
    embargo_days: int = Field(0, ge=0)  # gap between train and test
    model_type: Literal["random_forest", "logistic_regression", "gradient_boosting"] = (
        "random_forest"
    )
    model_params: dict[str, float | int | str] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Top-level experiment config — the entire pipeline."""

    name: str
    description: str = ""
    data: DataConfig
    backtest: BacktestConfig = BacktestConfig()
    strategy: StrategyConfig
    walkforward: WalkForwardConfig | None = None
    mlflow_tracking_uri: str = "file:./mlruns"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate a YAML config file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
