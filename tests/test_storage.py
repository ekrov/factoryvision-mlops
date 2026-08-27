"""Tests for prediction persistence."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine

from factoryvision.api.inference import Prediction
from factoryvision.api.schemas import BoundingBox
from factoryvision.storage.database import initialize_database
from factoryvision.storage.repository import PredictionRepository, image_id_from_bytes


def _repository(tmp_path: Path) -> PredictionRepository:
    engine = create_engine(f"sqlite:///{tmp_path / 'predictions.db'}")
    initialize_database(engine)
    return PredictionRepository.from_engine(engine)


def test_image_id_is_stable_and_content_based() -> None:
    first = image_id_from_bytes(b"same image")
    second = image_id_from_bytes(b"same image")
    different = image_id_from_bytes(b"different image")

    assert first == second
    assert first != different
    assert len(first) == 64


def test_repository_saves_success_with_bounding_box(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    prediction = Prediction(
        has_defect=True,
        defect_probability=0.91,
        defect_area_fraction=0.02,
        bounding_box=BoundingBox(x_min=1, y_min=2, x_max=10, y_max=20),
        mask_base64="encoded-mask",
        original_image_height=32,
        original_image_width=48,
        mask_height=256,
        mask_width=640,
    )

    record = repository.save_success(
        image_id="image-1",
        model_name="factoryvision-segmentation",
        model_alias="candidate",
        prediction=prediction,
        latency_ms=12.5,
    )

    stored = repository.latest_for_image("image-1")
    assert stored is not None
    assert stored.id == record.id
    assert stored.status == "success"
    assert stored.x_min == 1
    assert stored.y_max == 20
    assert stored.error_message is None


def test_repository_saves_failure_without_prediction_values(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    repository.save_failure(
        image_id="image-2",
        model_name="factoryvision-segmentation",
        model_alias="candidate",
        latency_ms=4.2,
        error_message="invalid image",
    )

    stored = repository.latest_for_image("image-2")
    assert stored is not None
    assert stored.status == "error"
    assert stored.defect_probability is None
    assert stored.defect_area_fraction is None
    assert stored.error_message == "invalid image"
