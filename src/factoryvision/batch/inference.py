"""Reusable batch-inference steps used by the Airflow DAG.

Airflow coordinates these functions, but the functions deliberately do not
depend on Airflow. This makes the batch behavior easy to test and run locally.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from factoryvision.api.inference import (
    REPO_ROOT,
    InferenceConfig,
    InvalidImageError,
    OnnxSegmenter,
)
from factoryvision.storage.database import (
    create_database_engine,
    database_url_from_environment,
    initialize_database,
)
from factoryvision.storage.repository import (
    PredictionRepository,
    image_id_from_bytes,
)


SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class BatchConfig:
    """Paths and limits for one scheduled batch run."""

    image_dir: Path
    artifact_dir: Path
    database_url: str
    max_images: int | None = None


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def batch_config_from_environment() -> BatchConfig:
    """Read batch settings without requiring Airflow to be installed locally."""

    raw_max_images = os.getenv("FACTORYVISION_BATCH_MAX_IMAGES", "")
    max_images = int(raw_max_images) if raw_max_images else None
    if max_images is not None and max_images <= 0:
        raise ValueError("FACTORYVISION_BATCH_MAX_IMAGES must be positive")
    return BatchConfig(
        image_dir=_resolve_path(
            os.getenv("FACTORYVISION_BATCH_IMAGE_DIR", "data/batch/incoming")
        ),
        artifact_dir=_resolve_path(
            os.getenv("FACTORYVISION_BATCH_ARTIFACT_DIR", "artifacts/batch")
        ),
        database_url=database_url_from_environment(),
        max_images=max_images,
    )


def _new_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def discover_new_images(
    image_dir: Path,
    repository: PredictionRepository,
    max_images: int | None = None,
) -> list[str]:
    """Find image files without an existing prediction record.

    The image content hash is used for identity, so moving or renaming an
    image does not make it look new to the batch job.
    """

    new_paths: list[str] = []
    for path in _new_image_paths(image_dir):
        try:
            image_id = image_id_from_bytes(path.read_bytes())
        except OSError:
            continue
        if repository.latest_for_image(image_id) is None:
            new_paths.append(str(path))
        if max_images is not None and len(new_paths) >= max_images:
            break
    return new_paths


def validate_image_paths(image_paths: Iterable[str]) -> list[str]:
    """Keep only readable images that OpenCV can decode."""

    valid_paths: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path)
        try:
            encoded = path.read_bytes()
        except OSError:
            continue
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None and image.size > 0:
            valid_paths.append(str(path))
    return valid_paths


def _bounding_box_dict(bounding_box: Any) -> dict[str, int] | None:
    if bounding_box is None:
        return None
    return {
        "x_min": int(bounding_box.x_min),
        "y_min": int(bounding_box.y_min),
        "x_max": int(bounding_box.x_max),
        "y_max": int(bounding_box.y_max),
    }


def _run_id_from_path(path: Path) -> str:
    return path.parent.name or "manual-run"


def run_inference_to_file(
    image_paths: Iterable[str],
    output_path: Path,
    config: InferenceConfig | None = None,
    segmenter: OnnxSegmenter | None = None,
) -> str:
    """Run inference and write compact prediction metadata to JSON.

    The mask is intentionally excluded from this hand-off file. It is large,
    is already returned by the online API, and is not needed by PostgreSQL.
    """

    inference_config = config or InferenceConfig.from_environment()
    active_segmenter = segmenter or OnnxSegmenter(inference_config)
    predictions: list[dict[str, Any]] = []
    for raw_path in image_paths:
        path = Path(raw_path)
        image_bytes = path.read_bytes()
        image_id = image_id_from_bytes(image_bytes)
        started_at = time.perf_counter()
        try:
            prediction = active_segmenter.predict(image_bytes)
        except (InvalidImageError, RuntimeError, ValueError) as error:
            predictions.append(
                {
                    "image_id": image_id,
                    "image_path": str(path),
                    "status": "error",
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                    "error_message": str(error),
                }
            )
            continue
        predictions.append(
            {
                "image_id": image_id,
                "image_path": str(path),
                "status": "success",
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                "defect_probability": prediction.defect_probability,
                "defect_area_fraction": prediction.defect_area_fraction,
                "has_defect": prediction.has_defect,
                "bounding_box": _bounding_box_dict(prediction.bounding_box),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "batch_run_id": _run_id_from_path(output_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_name": inference_config.model_name,
                "model_alias": inference_config.model_alias,
                "predictions": predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(output_path)


def persist_prediction_artifact(
    prediction_path: Path,
    config: BatchConfig | None = None,
) -> dict[str, Any]:
    """Persist the inference hand-off file and return run-level statistics."""

    batch_config = config or batch_config_from_environment()
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    engine = create_database_engine(batch_config.database_url)
    initialize_database(engine)
    repository = PredictionRepository.from_engine(engine)
    try:
        for item in payload["predictions"]:
            if item["status"] == "success":
                from factoryvision.api.schemas import BoundingBox

                bounding_box = (
                    BoundingBox(**item["bounding_box"])
                    if item["bounding_box"] is not None
                    else None
                )
                repository.save_success_metadata(
                    image_id=item["image_id"],
                    model_name=payload["model_name"],
                    model_alias=payload["model_alias"],
                    defect_probability=float(item["defect_probability"]),
                    defect_area_fraction=float(item["defect_area_fraction"]),
                    bounding_box=bounding_box,
                    latency_ms=float(item["latency_ms"]),
                )
            else:
                repository.save_failure(
                    image_id=item["image_id"],
                    model_name=payload["model_name"],
                    model_alias=payload["model_alias"],
                    latency_ms=float(item["latency_ms"]),
                    error_message=item["error_message"],
                )
    finally:
        engine.dispose()
    return _summary_from_payload(payload)


def _summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    predictions = payload["predictions"]
    successful = [item for item in predictions if item["status"] == "success"]
    failed = [item for item in predictions if item["status"] == "error"]
    latencies = [float(item["latency_ms"]) for item in predictions]
    return {
        "batch_run_id": payload["batch_run_id"],
        "created_at": payload["created_at"],
        "model_name": payload["model_name"],
        "model_alias": payload["model_alias"],
        "total_images": len(predictions),
        "successful_predictions": len(successful),
        "failed_predictions": len(failed),
        "defects_detected": sum(bool(item.get("has_defect")) for item in successful),
        "average_defect_probability": (
            sum(float(item["defect_probability"]) for item in successful) / len(successful)
            if successful
            else None
        ),
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


def write_summary_artifact(
    prediction_path: Path,
    persisted_summary: dict[str, Any],
    artifact_dir: Path,
) -> str:
    """Write the human- and machine-readable summary for one DAG run."""

    summary_path = artifact_dir / prediction_path.parent.name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                **persisted_summary,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "prediction_artifact": str(prediction_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(summary_path)
