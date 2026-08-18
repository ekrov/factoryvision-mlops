"""PyTorch dataset loader for the KolektorSDD2 dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import cv2
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


DatasetSplit = Literal["train", "validation", "test"]
ImageSize = tuple[int, int]


class KolektorSDD2Dataset(Dataset[tuple[Tensor, Tensor]]):
    """Load KolektorSDD2 image/mask pairs from the reusable split manifest.

    Images are returned as RGB float32 tensors with shape (3, height, width)
    and values in [0, 1]. Masks are returned as binary float32 tensors with
    shape (1, height, width) and values in {0, 1}.
    """

    def __init__(
        self,
        manifest_path: str | Path = "data/processed/splits.csv",
        split: DatasetSplit = "train",
        image_size: ImageSize = (640, 256),
        augmentations: dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.repo_root = self.manifest_path.parents[2]
        self.image_size = image_size
        self.augmentations = augmentations or {"enabled": False}

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        records = pd.read_csv(self.manifest_path)
        required_columns = {
            "image_path",
            "mask_path",
            "mask_exists",
            "dataset_split",
        }
        missing_columns = required_columns.difference(records.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.records = records[records["dataset_split"] == split].reset_index(drop=True)
        self.split = split

        if self.records.empty:
            raise ValueError(f"No records found for split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records.iloc[index]
        image_path = self.repo_root / str(record["image_path"])
        mask_path = self.repo_root / str(record["mask_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        target_height, target_width = self.image_size
        original_height, original_width = image.shape[:2]
        scale = min(
            target_height / original_height,
            target_width / original_width,
        )
        resized_height = min(target_height, max(1, round(original_height * scale)))
        resized_width = min(target_width, max(1, round(original_width * scale)))

        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Letterbox both arrays into the same target canvas. Image padding
        # reflects nearby texture; mask padding is background (zero).
        top = (target_height - resized_height) // 2
        left = (target_width - resized_width) // 2
        bottom = target_height - resized_height - top
        right = target_width - resized_width - left
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.copyMakeBorder(
            mask,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        image, mask = self._apply_augmentations(image, mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        image_tensor = image_tensor / 255.0

        mask_tensor = torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0)

        return image_tensor, mask_tensor

    def _apply_augmentations(
        self,
        image: Any,
        mask: Any,
    ) -> tuple[Any, Any]:
        """Apply configured spatial transforms to an image/mask pair."""

        if self.split != "train" or not self.augmentations.get("enabled", False):
            return image, mask

        horizontal_probability = float(
            self.augmentations.get("horizontal_flip_probability", 0.0)
        )
        vertical_probability = float(
            self.augmentations.get("vertical_flip_probability", 0.0)
        )
        for name, probability in (
            ("horizontal_flip_probability", horizontal_probability),
            ("vertical_flip_probability", vertical_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if random.random() < horizontal_probability:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < vertical_probability:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        return image, mask
