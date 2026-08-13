"""Metrics for binary defect-segmentation predictions."""

from __future__ import annotations

import torch
from torch import Tensor


def aggregate_metric_counts(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Return additive pixel- and image-level confusion counts."""

    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5

    predicted_defect = predictions.flatten(start_dim=1).any(dim=1)
    actual_defect = targets.flatten(start_dim=1).any(dim=1)

    return {
        "true_positive": int((predictions & targets).sum().item()),
        "false_positive": int((predictions & ~targets).sum().item()),
        "false_negative": int((~predictions & targets).sum().item()),
        "defect_true_positive": int((predicted_defect & actual_defect).sum().item()),
        "defect_false_positive": int((predicted_defect & ~actual_defect).sum().item()),
        "defect_false_negative": int((~predicted_defect & actual_defect).sum().item()),
    }


def merge_metric_counts(
    counts: dict[str, int],
    batch_counts: dict[str, int],
) -> None:
    """Add one batch of counts to a running total."""

    for key in counts:
        counts[key] += batch_counts[key]


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Calculate metrics from counts accumulated over a complete split."""

    true_positive = float(counts["true_positive"])
    false_positive = float(counts["false_positive"])
    false_negative = float(counts["false_negative"])
    defect_true_positive = float(counts["defect_true_positive"])
    defect_false_positive = float(counts["defect_false_positive"])
    defect_false_negative = float(counts["defect_false_negative"])

    return {
        "dice": (2 * true_positive)
        / max(2 * true_positive + false_positive + false_negative, 1.0),
        "iou": true_positive
        / max(true_positive + false_positive + false_negative, 1.0),
        "precision": true_positive / max(true_positive + false_positive, 1.0),
        "recall": true_positive / max(true_positive + false_negative, 1.0),
        "defect_f1": (2 * defect_true_positive)
        / max(
            2 * defect_true_positive
            + defect_false_positive
            + defect_false_negative,
            1.0,
        ),
    }


def segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate metrics for one batch."""

    counts = {
        key: 0
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "defect_true_positive",
            "defect_false_positive",
            "defect_false_negative",
        )
    }
    merge_metric_counts(counts, aggregate_metric_counts(logits, targets, threshold))
    return metrics_from_counts(counts)
