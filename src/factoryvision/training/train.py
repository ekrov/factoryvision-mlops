"""Training utilities for the FactoryVision U-Net baseline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

from factoryvision.data import KolektorSDD2Dataset

from .losses import BCEDiceLoss
from .metrics import aggregate_metric_counts, merge_metric_counts, metrics_from_counts
from .unet import UNet, UNetConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Small, explicit baseline-training configuration."""

    manifest_path: str | Path = "data/processed/splits.csv"
    image_size: tuple[int, int] = (640, 256)
    batch_size: int = 2
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    bce_weight: float = 0.25
    dice_weight: float = 0.75
    max_pos_weight: float = 50.0
    epochs: int = 5
    run_dir: str | Path = "artifacts/runs/baseline"
    seed: int = 42
    deterministic: bool = True
    device: str | None = "auto"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.run_dir) / "checkpoints" / "best.pt"

    @property
    def history_path(self) -> Path:
        return Path(self.run_dir) / "history.csv"

    @property
    def config_path(self) -> Path:
        return Path(self.run_dir) / "config.json"

    @property
    def preview_path(self) -> Path:
        return Path(self.run_dir) / "validation_previews.png"


def build_dataloaders_from_datasets(
    train_dataset: KolektorSDD2Dataset,
    validation_dataset: KolektorSDD2Dataset,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation loaders from already-loaded datasets."""

    set_seed(config.seed, config.deterministic)

    class_weights = train_dataset.records["class"].map(
        {"defect": 1.0, "non-defect": 0.0}
    )
    defect_count = float(class_weights.sum())
    non_defect_count = float(len(class_weights) - defect_count)
    if defect_count == 0:
        raise ValueError("Training split contains no defective samples.")
    sample_weights = class_weights.replace(
        {1.0: non_defect_count / defect_count, 0.0: 1.0}
    ).to_numpy()
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and validation loaders from the reusable manifest."""

    train_dataset = KolektorSDD2Dataset(
        manifest_path=config.manifest_path,
        split="train",
        image_size=config.image_size,
    )
    validation_dataset = KolektorSDD2Dataset(
        manifest_path=config.manifest_path,
        split="validation",
        image_size=config.image_size,
    )
    return build_dataloaders_from_datasets(train_dataset, validation_dataset, config)


def calculate_pos_weight(
    dataset: KolektorSDD2Dataset,
    max_pos_weight: float = 50.0,
) -> tuple[float, int, int]:
    """Measure training-pixel imbalance and return a bounded BCE weight."""

    positive_pixels = 0
    total_pixels = 0
    for mask_path in dataset.records["mask_path"]:
        mask = cv2.imread(
            str(dataset.repo_root / str(mask_path)),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        positive_pixels += int((mask > 0).sum())
        total_pixels += mask.size

    negative_pixels = total_pixels - positive_pixels
    raw_weight = negative_pixels / max(positive_pixels, 1)
    return min(raw_weight, max_pos_weight), positive_pixels, negative_pixels


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation epoch and return mean loss."""

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_function(logits, masks)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def _config_as_json(config: TrainingConfig) -> dict[str, Any]:
    values = asdict(config)
    values["image_size"] = list(config.image_size)
    values["manifest_path"] = str(config.manifest_path)
    values["run_dir"] = str(config.run_dir)
    return values


def _save_run_config(
    config: TrainingConfig,
    model_config: UNetConfig | None = None,
) -> None:
    config.config_path.parent.mkdir(parents=True, exist_ok=True)
    values = _config_as_json(config)
    if model_config is not None:
        values["model"] = {
            "in_channels": model_config.in_channels,
            "out_channels": model_config.out_channels,
            "base_channels": model_config.base_channels,
        }
    config.config_path.write_text(
        json.dumps(values, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or choose CUDA when it is available."""

    if device is None or str(device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from PyTorch's per-worker initial seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_model(model_config: UNetConfig | None = None) -> UNet:
    """Construct the U-Net from explicit architecture configuration."""

    selected_config = model_config or UNetConfig()
    return UNet(
        in_channels=selected_config.in_channels,
        out_channels=selected_config.out_channels,
        base_channels=selected_config.base_channels,
    )


def build_loss_function(
    dataset: KolektorSDD2Dataset,
    config: TrainingConfig,
    device: torch.device,
) -> nn.Module:
    """Create the configured segmentation loss using training-pixel imbalance."""

    pos_weight, positive_pixels, negative_pixels = calculate_pos_weight(
        dataset,
        max_pos_weight=config.max_pos_weight,
    )
    print(
        f"positive training pixels: {positive_pixels:,} | "
        f"negative training pixels: {negative_pixels:,} | "
        f"BCE pos_weight: {pos_weight:.2f}"
    )
    return BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32),
    ).to(device)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Evaluate loss and aggregate metrics across the full validation set."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    counts = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "defect_true_positive": 0,
        "defect_false_positive": 0,
        "defect_false_negative": 0,
    }

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_function(logits, masks)
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            merge_metric_counts(
                counts,
                aggregate_metric_counts(logits, masks, threshold),
            )

    return total_loss / total_samples, metrics_from_counts(counts)


def save_validation_preview(
    model: nn.Module,
    dataset: KolektorSDD2Dataset,
    device: torch.device,
    output_path: Path,
    examples_per_class: int = 2,
) -> None:
    """Save image, ground truth, and prediction for several samples."""

    model.eval()
    selected_indices = []
    for class_name in ("defect", "non-defect"):
        indices = dataset.records.index[
            dataset.records["class"] == class_name
        ].tolist()
        selected_indices.extend(indices[:examples_per_class])

    figure, axes = plt.subplots(
        len(selected_indices),
        3,
        figsize=(18, 5 * len(selected_indices)),
    )
    axes = np.atleast_2d(axes)

    for row_index, sample_index in enumerate(selected_indices):
        image, mask = dataset[sample_index]
        with torch.no_grad():
            probabilities = torch.sigmoid(model(image.unsqueeze(0).to(device)))

        image_for_display = image.permute(1, 2, 0).numpy()
        target = mask[0].numpy()
        prediction = (
            probabilities[0, 0].detach().cpu().numpy() >= 0.5
        ).astype(float)
        class_name = dataset.records.iloc[sample_index]["class"]

        axes[row_index, 0].imshow(image_for_display)
        axes[row_index, 0].set_title(f"{class_name} validation image")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Ground-truth mask")
        axes[row_index, 1].axis("off")
        axes[row_index, 2].imshow(image_for_display)
        axes[row_index, 2].imshow(prediction, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        axes[row_index, 2].set_title("Predicted defect overlay")
        axes[row_index, 2].axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a saved model checkpoint and optionally restore the optimizer."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_model_from_loaders(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    device: str | torch.device | None = None,
    model_config: UNetConfig | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train from Kedro-provided loaders and return the best model plus history."""

    set_seed(config.seed, config.deterministic)
    selected_device = resolve_device(device if device is not None else config.device)
    train_dataset = train_loader.dataset
    selected_model_config = model_config or UNetConfig()
    model = create_model(selected_model_config).to(selected_device)
    loss_function = build_loss_function(train_dataset, config, selected_device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    _save_run_config(config, selected_model_config)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")

    print(f"Training on: {selected_device}")
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            loss_function,
            selected_device,
            optimizer,
        )
        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            selected_device,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **validation_metrics,
        }
        history.append(row)
        _save_history(history, config.history_path)
        print(
            f"epoch {epoch:03d}/{config.epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"validation_loss={validation_loss:.4f} | "
            f"dice={validation_metrics['dice']:.4f} | "
            f"iou={validation_metrics['iou']:.4f} | "
            f"precision={validation_metrics['precision']:.4f} | "
            f"recall={validation_metrics['recall']:.4f} | "
            f"defect_f1={validation_metrics['defect_f1']:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "config": _config_as_json(config),
                },
                config.checkpoint_path,
            )
            print(f"saved checkpoint: {config.checkpoint_path}")

    load_checkpoint(model, config.checkpoint_path, selected_device)
    return model, history


def train_model(
    config: TrainingConfig = TrainingConfig(),
    device: str | torch.device | None = None,
) -> list[dict[str, float]]:
    """Train U-Net and persist history, configuration, checkpoint, and preview."""

    selected_device = resolve_device(device if device is not None else config.device)
    train_loader, validation_loader = build_dataloaders(config)
    model, history = train_model_from_loaders(
        train_loader,
        validation_loader,
        config,
        selected_device,
    )
    save_validation_preview(
        model,
        validation_loader.dataset,
        selected_device,
        config.preview_path,
    )
    print(f"saved validation previews: {config.preview_path}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--run-dir", default="artifacts/runs/baseline")
    args = parser.parse_args()
    train_model(
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            run_dir=args.run_dir,
        )
    )


if __name__ == "__main__":
    main()
