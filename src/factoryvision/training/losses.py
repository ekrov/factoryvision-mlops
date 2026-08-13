"""Loss functions for binary segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Soft Dice loss calculated from logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.float().flatten(start_dim=1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Average BCE-with-logits and soft Dice losses."""

    def __init__(
        self,
        bce_weight: float = 0.25,
        dice_weight: float = 0.75,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.float()
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
