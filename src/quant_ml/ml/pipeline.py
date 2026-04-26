"""ML pipeline construction.

Builds (features, labels) from raw price data and instantiates models from
config. Centralizing this prevents the train-prod skew problem: the same
function builds features for training, walk-forward testing, and live
inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from quant_ml.features.technical import build_feature_matrix


def make_classifier(model_type: str, params: dict[str, Any] | None = None) -> ClassifierMixin:
    """Factory for sklearn classifiers."""
    params = params or {}
    if model_type == "random_forest":
        defaults = {"n_estimators": 200, "max_depth": 5, "random_state": 42, "n_jobs": -1}
        return RandomForestClassifier(**{**defaults, **params})
    if model_type == "logistic_regression":
        defaults = {"max_iter": 1000, "random_state": 42}
        return LogisticRegression(**{**defaults, **params})
    if model_type == "gradient_boosting":
        defaults = {"n_estimators": 200, "max_depth": 3, "random_state": 42}
        return GradientBoostingClassifier(**{**defaults, **params})
    raise ValueError(f"Unknown model_type: {model_type}")


def build_dataset(
    prices: pd.DataFrame,
    horizon: int = 1,
    close_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for ML training.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLCV data.
    horizon : int
        Forecast horizon in bars. y_t = 1 if return from t to t+horizon > 0.
    close_col : str
        Price column to use for return calculation.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, NaN rows dropped.
    y : pd.Series
        Binary target aligned with X.
    """
    features = build_feature_matrix(prices, close_col=close_col)
    forward_return = prices[close_col].shift(-horizon) / prices[close_col] - 1
    target = (forward_return > 0).astype(int)
    target.name = "target"

    df = features.join(target).dropna()
    # Drop the last `horizon` rows where target is NaN (no future to look at)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Replace any inf from divisions with NaN, then drop
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    return X, y
