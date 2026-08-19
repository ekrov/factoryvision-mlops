"""Register the best mini-study model in the local MLflow Model Registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from factoryvision.training.train import create_model
from factoryvision.training.unet import UNetConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow.db"
DEFAULT_WINNER_PATH = REPO_ROOT / "artifacts" / "mini-study" / "winner.json"
DEFAULT_MODEL_NAME = "factoryvision-segmentation"
DEFAULT_ALIAS = "candidate"
INPUT_SHAPE = (1, 3, 256, 640)


def _load_winner(path: Path) -> dict[str, Any]:
    """Read the persisted mini-study winner and validate its run id."""

    if not path.exists():
        raise FileNotFoundError(
            f"Winner summary does not exist: {path}. "
            "Run scripts/run_mini_study.py first."
        )
    winner = json.loads(path.read_text(encoding="utf-8"))
    run_id = winner.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Winner summary has no valid run_id: {path}")
    return winner


def _find_existing_version(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> Any | None:
    """Return a registry version already created from this MLflow run."""

    versions = client.search_model_versions()
    matches = [
        version
        for version in versions
        if version.name == model_name and version.run_id == run_id
    ]
    if not matches:
        return None
    return max(matches, key=lambda version: int(version.version))


def _model_from_run(run: Any, checkpoint_path: Path) -> torch.nn.Module:
    """Rebuild the U-Net architecture and load the tracked best checkpoint."""

    params = run.data.params
    model_config = UNetConfig(
        in_channels=int(params["model.in_channels"]),
        out_channels=int(params["model.out_channels"]),
        base_channels=int(params["model.base_channels"]),
    )
    model = create_model(model_config).cpu()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _set_registry_metadata(
    client: MlflowClient,
    model_name: str,
    version: Any,
    winner: dict[str, Any],
    alias: str,
) -> None:
    """Attach searchable metadata and a human-readable model alias."""

    version_number = str(version.version)
    client.set_registered_model_alias(model_name, alias, version_number)
    client.set_registered_model_tag(model_name, "task", "binary defect segmentation")
    client.set_registered_model_tag(model_name, "selection_metric", "final_iou")
    client.set_model_version_tag(model_name, version_number, "study_name", "factoryvision-mini-study")
    client.set_model_version_tag(model_name, version_number, "configuration", str(winner["name"]))
    client.set_model_version_tag(model_name, version_number, "final_iou", str(winner["final_iou"]))
    client.set_model_version_tag(model_name, version_number, "final_defect_f1", str(winner["final_defect_f1"]))


def register_best_model(
    winner_path: Path = DEFAULT_WINNER_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_name: str = DEFAULT_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
) -> dict[str, str]:
    """Register the persisted winner and return its registry coordinates."""

    winner = _load_winner(winner_path)
    run_id = str(winner["run_id"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    existing = _find_existing_version(client, model_name, run_id)
    if existing is not None:
        _set_registry_metadata(client, model_name, existing, winner, alias)
        return {
            "model_name": model_name,
            "version": str(existing.version),
            "alias": alias,
            "run_id": run_id,
            "status": "already_registered",
        }

    checkpoint = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/best.pt",
        )
    )
    model = _model_from_run(run, checkpoint)
    input_example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        output_example = model(input_example)
    signature = infer_signature(
        input_example.numpy(),
        output_example.numpy(),
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(
            {
                "model_registry": model_name,
                "model_selection": "best mini-study final IoU",
            }
        )
        mlflow.pytorch.log_model(
            model,
            name="pytorch_model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example.numpy(),
            code_paths=[str(REPO_ROOT / "src")],
            pip_requirements=[
                "mlflow==3.14.0",
                "numpy==2.2.1",
                "torch==2.5.1",
            ],
            serialization_format="pickle",
        )

    version = _find_existing_version(client, model_name, run_id)
    if version is None:
        raise RuntimeError(
            f"MLflow logged the model but no registry version was found for run {run_id}."
        )
    _set_registry_metadata(client, model_name, version, winner, alias)
    return {
        "model_name": model_name,
        "version": str(version.version),
        "alias": alias,
        "run_id": run_id,
        "status": "registered",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER_PATH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    args = parser.parse_args()

    result = register_best_model(
        winner_path=args.winner,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        alias=args.alias,
    )
    print(json.dumps(result, indent=2))
    print(
        "Load the selected model with: "
        f"models:/{result['model_name']}@{result['alias']}"
    )


if __name__ == "__main__":
    main()
