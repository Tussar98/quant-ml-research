"""Command-line interface.

Usage:
    quant-ml backtest --config configs/default.yaml
    quant-ml walkforward --config configs/walkforward.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

from quant_ml.config import ExperimentConfig


@click.group()
def cli() -> None:
    """quant-ml: production-grade quantitative trading research framework."""
    pass


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML experiment config.",
)
@click.option("--no-mlflow", is_flag=True, help="Disable MLflow tracking.")
def backtest(config_path: Path, no_mlflow: bool) -> None:
    """Run a single backtest from a config file."""
    from scripts.run_backtest import main as run_main

    cfg = ExperimentConfig.from_yaml(config_path)
    logger.info(f"Loaded experiment: {cfg.name}")
    run_main(cfg, enable_mlflow=not no_mlflow)


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML experiment config.",
)
@click.option("--no-mlflow", is_flag=True, help="Disable MLflow tracking.")
def walkforward(config_path: Path, no_mlflow: bool) -> None:
    """Run walk-forward ML validation + backtest from a config file."""
    from scripts.run_walkforward import main as run_main

    cfg = ExperimentConfig.from_yaml(config_path)
    if cfg.walkforward is None:
        logger.error("Config has no `walkforward` section")
        sys.exit(1)
    logger.info(f"Loaded experiment: {cfg.name}")
    run_main(cfg, enable_mlflow=not no_mlflow)


if __name__ == "__main__":
    cli()
