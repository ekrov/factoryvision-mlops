"""Run a small, comparable FactoryVision hyperparameter study."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = "sqlite:///artifacts/mlflow.db"
EXPERIMENT_NAME = "factoryvision-mini-study"

# Keep the dataset, seed, image size, and batch size fixed. Each row changes
# one meaningful factor so the comparisons remain interpretable.
EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "baseline",
        "training.learning_rate": 0.001,
        "training.bce_weight": 0.25,
        "training.dice_weight": 0.75,
        "model.base_channels": 32,
    },
    {
        "name": "lower_learning_rate",
        "training.learning_rate": 0.0003,
        "training.bce_weight": 0.25,
        "training.dice_weight": 0.75,
        "model.base_channels": 32,
    },
    {
        "name": "balanced_loss",
        "training.learning_rate": 0.001,
        "training.bce_weight": 0.50,
        "training.dice_weight": 0.50,
        "model.base_channels": 32,
    },
    {
        "name": "smaller_unet",
        "training.learning_rate": 0.001,
        "training.bce_weight": 0.25,
        "training.dice_weight": 0.75,
        "model.base_channels": 16,
    },
)


def _runtime_parameters(experiment: dict[str, Any], epochs: int, device: str) -> str:
    """Build Kedro's comma-separated runtime parameter string."""

    values = {
        **experiment,
        "training.epochs": epochs,
        "training.seed": 42,
        "training.device": device,
        "training.run_dir": f"artifacts/runs/mini-study/{experiment['name']}",
        "tracking.tracking_uri": TRACKING_URI,
        "tracking.experiment_name": EXPERIMENT_NAME,
        "tracking.run_name": f"mini-study-{experiment['name']}",
    }
    return ",".join(f"{key}={value}" for key, value in values.items())


def _latest_run(client: MlflowClient, run_name: str) -> Any:
    """Find the completed run created for one experiment configuration."""

    runs = client.search_runs(
        experiment_ids=[
            mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        ],
        filter_string=f"tags.mlflow.runName = 'mini-study-{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"Could not find MLflow run for {run_name}")
    return runs[0]


def _run_experiment(experiment: dict[str, Any], epochs: int, device: str) -> dict[str, Any]:
    """Run one Kedro configuration and collect its MLflow results."""

    name = str(experiment["name"])
    params = _runtime_parameters(experiment, epochs, device)
    command = [
        sys.executable,
        "-m",
        "kedro",
        "run",
        "--pipelines",
        "baseline",
        "--params",
        params,
    ]
    print(f"\n=== Running {name} ===")
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    client = MlflowClient()
    run = _latest_run(client, name)
    metrics = run.data.metrics
    return {
        "name": name,
        "run_id": run.info.run_id,
        "learning_rate": experiment["training.learning_rate"],
        "bce_weight": experiment["training.bce_weight"],
        "dice_weight": experiment["training.dice_weight"],
        "base_channels": experiment["model.base_channels"],
        "final_iou": metrics.get("final_iou"),
        "final_dice": metrics.get("final_dice"),
        "final_precision": metrics.get("final_precision"),
        "final_recall": metrics.get("final_recall"),
        "final_defect_f1": metrics.get("final_defect_f1"),
        "final_validation_loss": metrics.get("final_validation_loss"),
    }


def _save_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Save a CSV comparison and JSON winner summary under artifacts."""

    output_dir = REPO_ROOT / "artifacts" / "mini-study"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    winner = max(
        rows,
        key=lambda row: (
            float(row["final_iou"]),
            float(row["final_defect_f1"]),
        ),
    )
    winner_path = output_dir / "winner.json"
    winner_path.write_text(json.dumps(winner, indent=2) + "\n", encoding="utf-8")
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if not 1 <= args.limit <= len(EXPERIMENTS):
        raise ValueError(f"--limit must be between 1 and {len(EXPERIMENTS)}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    rows = [
        _run_experiment(experiment, args.epochs, args.device)
        for experiment in EXPERIMENTS[: args.limit]
    ]
    winner = _save_summary(rows)

    print("\n=== Mini-study comparison ===")
    for row in rows:
        print(
            f"{row['name']}: IoU={row['final_iou']:.4f} | "
            f"Dice={row['final_dice']:.4f} | "
            f"defect_f1={row['final_defect_f1']:.4f}"
        )
    print(f"\nWinner by final IoU: {winner['name']} ({winner['run_id']})")
    print("Saved: artifacts/mini-study/results.csv")
    print("Saved: artifacts/mini-study/winner.json")


if __name__ == "__main__":
    main()
