"""MLflow experiment tracking wrapper.

Logs every backtest run with its config, parameters, metrics, and equity curve
artifact. This is the model registry / experiment tracking signal that
hiring managers look for in MLOps.

The wrapper makes MLflow optional — if it's not installed or the tracking URI
is unreachable, runs fall back to a no-op so the rest of the pipeline isn't
blocked. Production-quality observability shouldn't break the critical path.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Thin wrapper around MLflow for backtest experiment tracking."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "file:./mlruns",
        enabled: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled and MLFLOW_AVAILABLE

        if self.enabled:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking: {tracking_uri} / {experiment_name}")
        elif enabled and not MLFLOW_AVAILABLE:
            logger.warning("MLflow not installed; experiment tracking disabled")

    @contextmanager
    def run(self, run_name: str | None = None) -> Iterator[Any]:
        """Context manager for a single tracked run.

        Usage:
            with tracker.run('baseline'):
                tracker.log_params({...})
                tracker.log_metrics({...})
        """
        if not self.enabled:
            yield None
            return

        with mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log strategy/config parameters."""
        if not self.enabled:
            return
        # MLflow requires str/numeric values; stringify dicts and lists
        flat = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in params.items()}
        mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log performance metrics."""
        if not self.enabled:
            return
        # Filter to numeric only
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric)

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        """Log a dict as a JSON artifact."""
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            path.write_text(json.dumps(data, indent=2, default=str))
            mlflow.log_artifact(str(path))

    def log_figure(self, fig: Any, filename: str) -> None:
        """Log a matplotlib figure as an artifact."""
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            fig.savefig(path, bbox_inches="tight", dpi=120)
            mlflow.log_artifact(str(path))

    def log_dataframe(self, df: Any, filename: str) -> None:
        """Log a DataFrame as a CSV artifact."""
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            df.to_csv(path)
            mlflow.log_artifact(str(path))
