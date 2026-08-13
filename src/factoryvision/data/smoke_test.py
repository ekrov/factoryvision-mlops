"""Run a small dataset-loader smoke test and save a visual example."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from .dataset import KolektorSDD2Dataset


def run_smoke_test(
    manifest_path: str | Path = "data/processed/splits.csv",
    output_path: str | Path = "src/factoryvision/data/dataset_smoke_test.png",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one defective sample, print tensor details, and save a visualization."""

    dataset = KolektorSDD2Dataset(manifest_path=manifest_path, split="train")
    defect_indices = dataset.records.index[dataset.records["class"] == "defect"].tolist()

    if not defect_indices:
        raise ValueError("The selected split contains no defective samples.")

    sample_index = defect_indices[0]
    image, mask = dataset[sample_index]
    record = dataset.records.iloc[sample_index]

    print("Smoke test sample:", record["image_path"])
    print("image shape:", tuple(image.shape))
    print("mask shape:", tuple(mask.shape))
    print("image dtype:", image.dtype)
    print("mask dtype:", mask.dtype)
    print("image value range:", (float(image.min()), float(image.max())))
    print("mask unique values:", torch.unique(mask).tolist())
    print("mask defect pixels:", int(mask.sum().item()))

    image_for_display = image.permute(1, 2, 0).numpy()
    mask_for_display = mask.squeeze(0).numpy()

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(image_for_display)
    axes[0].set_title("Resized RGB image")
    axes[0].axis("off")

    axes[1].imshow(mask_for_display, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Resized binary ground truth")
    axes[1].axis("off")

    axes[2].imshow(image_for_display)
    axes[2].imshow(mask_for_display, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Aligned overlay")
    axes[2].axis("off")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print("visualization saved to:", output_path)

    return image, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/splits.csv",
        help="Path to the reusable split manifest.",
    )
    parser.add_argument(
        "--output",
        default="src/factoryvision/data/dataset_smoke_test.png",
        help="Path for the image/mask/overlay visualization.",
    )
    args = parser.parse_args()
    run_smoke_test(manifest_path=args.manifest, output_path=args.output)


if __name__ == "__main__":
    main()
