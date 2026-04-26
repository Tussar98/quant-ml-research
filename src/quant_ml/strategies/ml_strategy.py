"""ML-based strategy that wraps a trained sklearn classifier.

The classifier predicts P(up) for the next bar. Long when P(up) > threshold,
flat otherwise. This separates the model (trained offline, walk-forward) from
the strategy (used at backtest/inference time).
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import ClassifierMixin

from quant_ml.features.technical import build_feature_matrix
from quant_ml.strategies.base import Strategy


class MLClassifierStrategy(Strategy):
    """Trade based on a sklearn classifier's probability of an up move.

    The model must be fitted before passing to this strategy. For walk-forward
    backtesting, see `quant_ml.ml.walkforward.WalkForwardValidator` which
    handles refit/predict windowing.
    """

    name = "ml_classifier"

    def __init__(
        self,
        model: ClassifierMixin,
        feature_cols: list[str] | None = None,
        threshold: float = 0.5,
        close_col: str = "Adj Close",
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.model = model
        self.feature_cols = feature_cols
        self.threshold = threshold
        self.close_col = close_col

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        features = build_feature_matrix(prices, close_col=self.close_col)

        # Drop rows with NaN features (warmup period for indicators)
        features_clean = features.dropna()
        cols = self.feature_cols or features_clean.columns.tolist()

        X = features_clean[cols].to_numpy()

        # Get probability of positive class
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba (e.g., raw SVC)
            probas = self.model.predict(X).astype(float)

        # Long when P(up) > threshold
        signals = (probas > self.threshold).astype(int)

        # Reindex back to original price index, fill warmup with 0
        signal_series = pd.Series(signals, index=features_clean.index, dtype=float)
        signal_series = signal_series.reindex(prices.index).fillna(0)

        # Lag by 1 bar to avoid look-ahead
        return signal_series.shift(1).fillna(0)
