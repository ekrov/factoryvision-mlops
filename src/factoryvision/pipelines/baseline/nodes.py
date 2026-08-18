"""Kedro nodes that adapt the existing FactoryVision baseline code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from torch import nn
from torch.utils.data import DataLoader

from factoryvision.data import KolektorSDD2Dataset
from factoryvision.training.train import (
    TrainingConfig,
    build_dataloaders_from_datasets,
    build_loss_function,
    evaluate,
    resolve_device,
    save_validation_preview,
    train_model_from_loaders,
)
from factoryvision.training.unet import UNetConfig
from factoryvision.tracking.mlflow_tracking import (
    configure_tracking,
    log_parameters,
    log_required_artifact,
    log_training_history,
)


def _training_config(
    dataset_params: dict[str, Any],
    training_params: dict[str, Any],
) -> TrainingConfig:
    """Combine Kedro dataset and training parameters into the baseline config."""

    values = {**dataset_params, **training_params}
    values["image_size"] = tuple(values["image_size"])
    return TrainingConfig(**values)


def _model_config(model_params: dict[str, Any]) -> UNetConfig:
    """Create the U-Net architecture configuration from Kedro parameters."""

    return UNetConfig(**model_params)


def ingest_datasets(
    dataset_params: dict[str, Any],
    augmentation_params: dict[str, Any],
) -> tuple[KolektorSDD2Dataset, KolektorSDD2Dataset]:
    """Create the train and validation dataset objects from the split manifest."""

    manifest_path = dataset_params["manifest_path"]
    image_size = tuple(dataset_params["image_size"])
    return (
        KolektorSDD2Dataset(
            manifest_path,
            split="train",
            image_size=image_size,
            augmentations=augmentation_params,
        ),
        KolektorSDD2Dataset(
            manifest_path,
            split="validation",
            image_size=image_size,
            augmentations={"enabled": False},
        ),
    )


def prepare_dataloaders(
    train_dataset: KolektorSDD2Dataset,
    validation_dataset: KolektorSDD2Dataset,
    dataset_params: dict[str, Any],
    training_params: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Build the train and validation DataLoaders used by the baseline."""

    config = _training_config(dataset_params, training_params)
    return build_dataloaders_from_datasets(train_dataset, validation_dataset, config)


def train_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    dataset_params: dict[str, Any],
    training_params: dict[str, Any],
    model_params: dict[str, Any],
    augmentation_params: dict[str, Any],
    tracking_params: dict[str, Any],
) -> tuple[nn.Module, list[dict[str, float]], str, str]:
    """Train U-Net using Kedro inputs and expose the run artifacts."""

    config = _training_config(dataset_params, training_params)
    model_config = _model_config(model_params)
    configure_tracking(tracking_params)
    with mlflow.start_run(run_name=str(tracking_params["run_name"])) as run:
        mlflow.set_tags(
            {
                "project": "FactoryVision",
                "pipeline": "baseline",
                "dataset": "KolektorSDD2",
            }
        )
        log_parameters(
            {"dataset": dataset_params},
            {"augmentation": augmentation_params},
            {"model": model_params},
            {"training": training_params},
        )
        model, history = train_model_from_loaders(
            train_loader,
            validation_loader,
            config,
            model_config=model_config,
        )
        log_training_history(history)
        log_required_artifact(config.config_path, "training")
        log_required_artifact(config.history_path, "training")
        log_required_artifact(config.checkpoint_path, "model")
        run_id = run.info.run_id

    return model, history, str(config.checkpoint_path), run_id


def evaluate_model(
    model: nn.Module,
    train_dataset: KolektorSDD2Dataset,
    validation_dataset: KolektorSDD2Dataset,
    validation_loader: DataLoader,
    dataset_params: dict[str, Any],
    training_params: dict[str, Any],
    evaluation_params: dict[str, Any],
    tracking_params: dict[str, Any],
    mlflow_run_id: str,
) -> tuple[dict[str, float], str]:
    """Evaluate the trained model and save qualitative validation overlays."""

    config = _training_config(dataset_params, training_params)
    device = resolve_device(config.device)
    model = model.to(device)
    loss_function = build_loss_function(train_dataset, config, device)
    validation_loss, metrics = evaluate(
        model,
        validation_loader,
        loss_function,
        device,
        threshold=float(evaluation_params["threshold"]),
    )
    metrics = {"validation_loss": validation_loss, **metrics}
    save_validation_preview(
        model,
        validation_dataset,
        device,
        config.preview_path,
        examples_per_class=int(evaluation_params["examples_per_class"]),
    )
    configure_tracking(tracking_params)
    with mlflow.start_run(run_id=mlflow_run_id):
        log_parameters({"evaluation": evaluation_params})
        mlflow.log_metrics(
            {f"final_{key}": float(value) for key, value in metrics.items()}
        )
        log_required_artifact(config.preview_path, "predictions")
    return metrics, str(Path(config.preview_path))
