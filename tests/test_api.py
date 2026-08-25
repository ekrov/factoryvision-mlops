"""Tests for the FactoryVision FastAPI service."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from factoryvision.api.inference import InvalidImageError, InferenceConfig, Prediction
from factoryvision.api.main import create_app
from factoryvision.api.schemas import BoundingBox, PredictionResponse
from factoryvision.storage.database import initialize_database
from factoryvision.storage.repository import PredictionRepository, image_id_from_bytes


class FakeSegmenter:
    """Deterministic test double that avoids loading the real ONNX model."""

    runtime_name = "onnxruntime"

    def __init__(self, has_defect: bool = True) -> None:
        self.config = InferenceConfig(model_path=Path("fake.onnx"))
        self.has_defect = has_defect

    def model_info(self) -> dict[str, object]:
        return {
            "model_name": self.config.model_name,
            "model_alias": self.config.model_alias,
            "runtime": self.runtime_name,
            "input_shape": [1, 3, 256, 640],
            "output_shape": [1, 1, 256, 640],
            "threshold": self.config.threshold,
        }

    def predict(self, image_bytes: bytes) -> Prediction:
        del image_bytes
        mask = np.zeros((256, 640), dtype=np.uint8)
        if self.has_defect:
            mask[10:30, 20:50] = 255
        success, encoded = cv2.imencode(".png", mask)
        assert success
        return Prediction(
            has_defect=self.has_defect,
            defect_probability=0.91 if self.has_defect else 0.1,
            defect_area_fraction=float((mask > 0).mean()),
            bounding_box=(
                BoundingBox(
                    x_min=20,
                    y_min=10,
                    x_max=49,
                    y_max=29,
                )
                if self.has_defect
                else None
            ),
            mask_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
            original_image_height=32,
            original_image_width=48,
            mask_height=256,
            mask_width=640,
        )

    def prediction_response(self, prediction: Prediction):
        return PredictionResponse(
            model_name=self.config.model_name,
            model_alias=self.config.model_alias,
            runtime=self.runtime_name,
            has_defect=prediction.has_defect,
            defect_probability=prediction.defect_probability,
            defect_area_fraction=prediction.defect_area_fraction,
            bounding_box=prediction.bounding_box,
            mask_base64=prediction.mask_base64,
            mask_media_type="image/png",
            mask_height=prediction.mask_height,
            mask_width=prediction.mask_width,
            original_image_height=prediction.original_image_height,
            original_image_width=prediction.original_image_width,
        )


def _image_upload() -> tuple[str, bytes, str]:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return "sample.png", encoded.tobytes(), "image/png"


def _prediction_store(tmp_path) -> PredictionRepository:
    engine = create_engine(f"sqlite:///{tmp_path / 'predictions.db'}")
    initialize_database(engine)
    return PredictionRepository.from_engine(engine)


def _metric_sample(metrics: str, name: str, **labels: str) -> str | None:
    """Find a metric sample without depending on label serialization order."""

    for line in metrics.splitlines():
        if line.startswith(f"{name}{{") and all(
            f'{key}="{value}"' in line for key, value in labels.items()
        ):
            return line
    return None


def test_health_and_model_info(tmp_path) -> None:
    store = _prediction_store(tmp_path)
    with TestClient(
        create_app(segmenter=FakeSegmenter(), prediction_store=store)
    ) as client:
        health = client.get("/health")
        info = client.get("/model-info")

        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model_loaded": True,
            "storage_ready": True,
        }
        assert info.status_code == 200
        assert info.json()["runtime"] == "onnxruntime"
        assert info.json()["input_shape"] == [1, 3, 256, 640]


def test_metrics_endpoint_exposes_prometheus_format(tmp_path) -> None:
    store = _prediction_store(tmp_path)
    with TestClient(
        create_app(segmenter=FakeSegmenter(), prediction_store=store)
    ) as client:
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "factoryvision_http_requests_total" in response.text


def test_predict_returns_mask_score_and_bounding_box(tmp_path) -> None:
    store = _prediction_store(tmp_path)
    upload = _image_upload()
    with TestClient(
        create_app(segmenter=FakeSegmenter(), prediction_store=store)
    ) as client:
        response = client.post("/predict", files={"file": upload})

        assert response.status_code == 200
        body = response.json()
        assert body["has_defect"] is True
        assert body["defect_probability"] == 0.91
        assert body["bounding_box"] == {
            "x_min": 20,
            "y_min": 10,
            "x_max": 49,
            "y_max": 29,
        }
        assert body["mask_media_type"] == "image/png"
        assert body["mask_base64"]
        record = store.latest_for_image(image_id_from_bytes(upload[1]))
        assert record is not None
        assert record.status == "success"
        assert record.model_name == "factoryvision-segmentation"
        assert record.defect_probability == 0.91
        assert record.latency_ms >= 0.0
        metrics = client.get("/metrics").text
        assert "factoryvision_inference_duration_seconds_count" in metrics
        assert _metric_sample(
            metrics,
            "factoryvision_predictions_total",
            outcome="defect",
            model_name="factoryvision-segmentation",
            model_alias="candidate",
        ) is not None


def test_metrics_record_no_defect_prediction(tmp_path) -> None:
    store = _prediction_store(tmp_path)
    with TestClient(
        create_app(
            segmenter=FakeSegmenter(has_defect=False),
            prediction_store=store,
        )
    ) as client:
        response = client.post("/predict", files={"file": _image_upload()})

        assert response.status_code == 200
        assert response.json()["has_defect"] is False
        metrics = client.get("/metrics").text
        assert _metric_sample(
            metrics,
            "factoryvision_predictions_total",
            outcome="no_defect",
            model_name="factoryvision-segmentation",
            model_alias="candidate",
        ) is not None


def test_predict_rejects_non_image_content_type(tmp_path) -> None:
    store = _prediction_store(tmp_path)
    with TestClient(
        create_app(segmenter=FakeSegmenter(), prediction_store=store)
    ) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 415


def test_metrics_record_prediction_error(tmp_path) -> None:
    class ErrorSegmenter(FakeSegmenter):
        def predict(self, image_bytes: bytes) -> Prediction:
            del image_bytes
            raise InvalidImageError("test image decoding failure")

    store = _prediction_store(tmp_path)
    with TestClient(
        create_app(segmenter=ErrorSegmenter(), prediction_store=store)
    ) as client:
        response = client.post("/predict", files={"file": _image_upload()})

        assert response.status_code == 400
        metrics = client.get("/metrics").text
        assert _metric_sample(
            metrics,
            "factoryvision_http_errors_total",
            method="POST",
            path="/predict",
            status="400",
        ) is not None
        assert _metric_sample(
            metrics,
            "factoryvision_predictions_total",
            outcome="error",
            model_name="factoryvision-segmentation",
            model_alias="candidate",
        ) is not None


def test_predict_rejects_oversized_upload(tmp_path) -> None:
    segmenter = FakeSegmenter()
    segmenter.config = InferenceConfig(
        model_path=Path("fake.onnx"),
        max_upload_bytes=4,
    )
    store = _prediction_store(tmp_path)
    with TestClient(
        create_app(segmenter=segmenter, prediction_store=store)
    ) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", b"12345", "image/png")},
        )

        assert response.status_code == 413
