"""Tests for the Airflow-independent batch-inference functions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from factoryvision.api.inference import InferenceConfig, Prediction
from factoryvision.api.schemas import BoundingBox
from factoryvision.batch import (
    BatchConfig,
    discover_new_images,
    persist_prediction_artifact,
    run_inference_to_file,
    validate_image_paths,
    write_summary_artifact,
)
from factoryvision.storage.database import create_database_engine, initialize_database
from factoryvision.storage.repository import PredictionRepository, image_id_from_bytes


def _repository(tmp_path: Path) -> PredictionRepository:
    engine = create_database_engine(f"sqlite:///{tmp_path / 'predictions.db'}")
    initialize_database(engine)
    return PredictionRepository.from_engine(engine)


def _write_image(path: Path) -> bytes:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 80
    success, encoded = cv2.imencode(".png", image)
    assert success
    path.write_bytes(encoded.tobytes())
    return encoded.tobytes()


class _FakeSegmenter:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def predict(self, image_bytes: bytes) -> Prediction:
        return Prediction(
            has_defect=True,
            defect_probability=0.8,
            defect_area_fraction=0.1,
            bounding_box=BoundingBox(x_min=1, y_min=2, x_max=4, y_max=5),
            mask_base64="not-needed-by-batch-handoff",
            original_image_height=12,
            original_image_width=16,
            mask_height=256,
            mask_width=640,
        )


def test_discovery_uses_content_id_and_validation_filters_bad_files(tmp_path: Path) -> None:
    image_path = tmp_path / "inspection.png"
    image_bytes = _write_image(image_path)
    (tmp_path / "not-an-image.png").write_text("invalid", encoding="utf-8")
    repository = _repository(tmp_path)

    assert discover_new_images(tmp_path, repository) == [
        str(image_path),
        str(tmp_path / "not-an-image.png"),
    ]
    assert validate_image_paths([str(image_path), str(tmp_path / "not-an-image.png")]) == [
        str(image_path)
    ]

    repository.save_failure(
        image_id=image_id_from_bytes(image_bytes),
        model_name="factoryvision-segmentation",
        model_alias="candidate",
        latency_ms=1.0,
        error_message="previous run",
    )
    assert discover_new_images(tmp_path, repository) == [str(tmp_path / "not-an-image.png")]


def test_batch_handoff_persists_metadata_and_writes_summary(tmp_path: Path) -> None:
    image_path = tmp_path / "inspection.png"
    _write_image(image_path)
    prediction_path = tmp_path / "batch-1" / "predictions.json"
    model_config = InferenceConfig(
        model_path=tmp_path / "unused.onnx",
        model_name="factoryvision-segmentation",
        model_alias="candidate",
    )
    run_inference_to_file(
        [str(image_path)],
        prediction_path,
        config=model_config,
        segmenter=_FakeSegmenter(model_config),
    )

    config = BatchConfig(
        image_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        database_url=f"sqlite:///{tmp_path / 'batch.db'}",
    )
    persisted = persist_prediction_artifact(prediction_path, config=config)
    summary_path = write_summary_artifact(prediction_path, persisted, config.artifact_dir)

    assert persisted["total_images"] == 1
    assert persisted["defects_detected"] == 1
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert summary["successful_predictions"] == 1
