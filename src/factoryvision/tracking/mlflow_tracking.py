"""Small MLflow helpers used by the FactoryVision Kedro pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import mlflow


def configure_tracking(tracking_params: Mapping[str, Any]) -> str:
    """Configure MLflow and return the normalized tracking URI."""

    tracking_uri = str(tracking_params["tracking_uri"])
    if not tracking_uri.startswith(
        ("file:", "http:", "https:", "sqlite:", "postgresql:", "mysql:")
    ):
        tracking_uri = Path(tracking_uri).resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(tracking_params["experiment_name"]))
    return tracking_uri


def flatten_parameters(
    values: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    """Flatten nested YAML settings into MLflow-compatible parameter values."""

    flattened: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_parameters(value, name))
        elif isinstance(value, (list, tuple)):
            flattened[name] = json.dumps(value)
        else:
            flattened[name] = "null" if value is None else str(value)
    return flattened


def log_parameters(*parameter_groups: Mapping[str, Any]) -> None:
    """Log several nested configuration groups under stable parameter names."""

    combined: dict[str, Any] = {}
    for group in parameter_groups:
        combined.update(group)
    mlflow.log_params(flatten_parameters(combined))


def log_training_history(history: list[dict[str, float]]) -> None:
    """Log one MLflow metric point for each training epoch."""

    for row in history:
        epoch = int(row["epoch"])
        metrics = {
            key: float(value)
            for key, value in row.items()
            if key != "epoch"
        }
        mlflow.log_metrics(metrics, step=epoch)


def log_required_artifact(path: str | Path, artifact_path: str) -> None:
    """Log an artifact and fail clearly if the training output is missing."""

    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"Expected MLflow artifact was not created: {artifact}")
    mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
