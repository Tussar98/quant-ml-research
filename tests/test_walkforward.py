"""Tests for walk-forward validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from quant_ml.ml.walkforward import WalkForwardValidator


class TestWalkForwardValidator:
    def test_invalid_n_splits(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardValidator(n_splits=1)

    def test_no_lookahead_in_folds(self) -> None:
        """Every test_start must be >= train_end (causality)."""
        idx = pd.bdate_range("2015-01-01", "2022-12-31")
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(0, 1, (len(idx), 3)), index=idx, columns=["a", "b", "c"])

        validator = WalkForwardValidator(n_splits=4, train_months=24, test_months=6)
        for fold in validator.split(X):
            assert fold.test_start >= fold.train_end

    def test_train_set_grows_or_stays(self) -> None:
        """Expanding-window: training set should grow over folds."""
        idx = pd.bdate_range("2015-01-01", "2022-12-31")
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(0, 1, (len(idx), 3)), index=idx, columns=["a", "b", "c"])

        validator = WalkForwardValidator(n_splits=4, train_months=24, test_months=6)
        sizes = [len(fold.train_idx) for fold in validator.split(X)]
        # Sizes must be monotonically non-decreasing
        assert all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1))

    def test_evaluate_returns_predictions_for_all_test_indices(self) -> None:
        idx = pd.bdate_range("2015-01-01", "2022-12-31")
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(0, 1, (len(idx), 3)), index=idx, columns=["a", "b", "c"])
        # Synthetic target with some signal so training is meaningful
        y = pd.Series((X["a"] + rng.normal(0, 0.5, len(idx)) > 0).astype(int), index=idx)

        validator = WalkForwardValidator(n_splits=3, train_months=24, test_months=6)
        result = validator.evaluate(LogisticRegression(max_iter=1000), X, y)

        assert len(result.predictions) > 0
        assert len(result.predictions) == len(result.probabilities)
        assert result.predictions.index.is_monotonic_increasing

    def test_evaluate_metrics_within_bounds(self) -> None:
        idx = pd.bdate_range("2015-01-01", "2022-12-31")
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (len(idx), 3)), index=idx, columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, len(idx)), index=idx)

        validator = WalkForwardValidator(n_splits=3, train_months=24, test_months=6)
        result = validator.evaluate(LogisticRegression(max_iter=1000), X, y)

        # Accuracy must be in [0, 1] for every fold
        assert ((result.fold_metrics["accuracy"] >= 0) & (result.fold_metrics["accuracy"] <= 1)).all()

    def test_rejects_non_datetime_index(self) -> None:
        X = pd.DataFrame(np.random.rand(100, 2), columns=["a", "b"])
        validator = WalkForwardValidator(n_splits=3)
        with pytest.raises(TypeError):
            list(validator.split(X))
