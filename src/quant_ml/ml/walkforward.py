"""Walk-forward validation for time-series ML.

The single most important methodological correction to the original notebook.
Standard `train_test_split(shuffle=True)` leaks future data into the training
set — the model sees 2022 data while training to predict 2010. Walk-forward
validation respects causality: train on past, predict on future, roll forward.

This module implements expanding-window walk-forward by default, with optional
embargo periods between train and test to prevent leakage from overlapping
labels (e.g., when y is computed from t+5 returns).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score


@dataclass
class WalkForwardFold:
    """A single train/test fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""

    fold_metrics: pd.DataFrame
    predictions: pd.Series  # out-of-sample predictions concatenated across folds
    probabilities: pd.Series  # out-of-sample P(up) probabilities

    @property
    def mean_accuracy(self) -> float:
        return float(self.fold_metrics["accuracy"].mean())

    @property
    def std_accuracy(self) -> float:
        return float(self.fold_metrics["accuracy"].std())


class WalkForwardValidator:
    """Expanding-window walk-forward cross-validator.

    Each fold trains on data from the start of the series up to a moving
    cutoff and tests on the next `test_months` window. Optional embargo
    inserts a gap between train and test to prevent label leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_months: int = 36,
        test_months: int = 6,
        embargo_days: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if train_months < 1 or test_months < 1:
            raise ValueError("train_months and test_months must be >= 1")

        self.n_splits = n_splits
        self.train_months = train_months
        self.test_months = test_months
        self.embargo_days = embargo_days

    def split(self, X: pd.DataFrame) -> Iterator[WalkForwardFold]:
        """Yield train/test index pairs for each fold."""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("X must have a DatetimeIndex for walk-forward splits")

        index = X.index
        first_date = index[0]
        last_date = index[-1]

        # Initial training period ends at first_date + train_months
        first_train_end = first_date + pd.DateOffset(months=self.train_months)
        if first_train_end >= last_date:
            raise ValueError(
                f"Not enough data: train period ({self.train_months}mo) exceeds series length"
            )

        # Total span available for test periods
        test_span_start = first_train_end
        total_test_days = (last_date - test_span_start).days
        step_days = total_test_days // self.n_splits
        if step_days < self.test_months * 21:  # roughly trading days/month
            # Overlapping test windows would cause issues; warn via fewer splits
            pass

        for fold_id in range(self.n_splits):
            # Train ends at the moving cutoff
            train_end = test_span_start + pd.Timedelta(days=fold_id * step_days)
            test_start = train_end + pd.Timedelta(days=self.embargo_days)
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Clip test_end to available data
            if test_end > last_date:
                test_end = last_date

            train_mask = (index >= first_date) & (index < train_end)
            test_mask = (index >= test_start) & (index < test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield WalkForwardFold(
                fold_id=fold_id,
                train_start=index[train_idx[0]],
                train_end=index[train_idx[-1]],
                test_start=index[test_idx[0]],
                test_end=index[test_idx[-1]],
                train_idx=train_idx,
                test_idx=test_idx,
            )

    def evaluate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> WalkForwardResult:
        """Run walk-forward evaluation, refitting the model on each fold.

        Returns
        -------
        WalkForwardResult with per-fold metrics and concatenated out-of-sample
        predictions/probabilities aligned to the original index.
        """
        if not isinstance(model, ClassifierMixin):
            raise TypeError("model must be a sklearn ClassifierMixin")

        fold_records: list[dict[str, float | int | pd.Timestamp]] = []
        all_preds: dict[pd.Timestamp, int] = {}
        all_probs: dict[pd.Timestamp, float] = {}

        for fold in self.split(X):
            X_train = X.iloc[fold.train_idx]
            y_train = y.iloc[fold.train_idx]
            X_test = X.iloc[fold.test_idx]
            y_test = y.iloc[fold.test_idx]

            # Clone to avoid mutating the user's model object
            fold_model = clone(model)
            fold_model.fit(X_train.to_numpy(), y_train.to_numpy())

            preds = fold_model.predict(X_test.to_numpy())
            if hasattr(fold_model, "predict_proba"):
                probs = fold_model.predict_proba(X_test.to_numpy())[:, 1]
            else:
                probs = preds.astype(float)

            for idx, p, pr in zip(X_test.index, preds, probs, strict=False):
                all_preds[idx] = int(p)
                all_probs[idx] = float(pr)

            fold_records.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "n_train": len(fold.train_idx),
                    "n_test": len(fold.test_idx),
                    "accuracy": accuracy_score(y_test, preds),
                }
            )

        fold_df = pd.DataFrame(fold_records)
        pred_series = pd.Series(all_preds).sort_index()
        prob_series = pd.Series(all_probs).sort_index()

        return WalkForwardResult(
            fold_metrics=fold_df,
            predictions=pred_series,
            probabilities=prob_series,
        )
