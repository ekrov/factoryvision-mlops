"""Tests for the FactoryVision FastAPI service."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from factoryvision.api.inference import InferenceConfig, Prediction
from factoryvision.api.main import create_app
from factoryvision.api.schemas import BoundingBox, PredictionResponse


class FakeSegmenter:
    """Deterministic test double that avoids loading the real ONNX model."""

    runtime_name = "onnxruntime"

    def __init__(self) -> None:
        self.config = InferenceConfig(model_path=Path("fake.onnx"))

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
        mask[10:30, 20:50] = 255
        success, encoded = cv2.imencode(".png", mask)
        assert success
        return Prediction(
            has_defect=True,
            defect_probability=0.91,
            defect_area_fraction=float((mask > 0).mean()),
            bounding_box=BoundingBox(
                x_min=20,
                y_min=10,
                x_max=49,
                y_max=29,
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


def test_health_and_model_info() -> None:
    with TestClient(create_app(segmenter=FakeSegmenter())) as client:
        health = client.get("/health")
        info = client.get("/model-info")

        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}
        assert info.status_code == 200
        assert info.json()["runtime"] == "onnxruntime"
        assert info.json()["input_shape"] == [1, 3, 256, 640]


def test_predict_returns_mask_score_and_bounding_box() -> None:
    with TestClient(create_app(segmenter=FakeSegmenter())) as client:
        response = client.post("/predict", files={"file": _image_upload()})

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


def test_predict_rejects_non_image_content_type() -> None:
    with TestClient(create_app(segmenter=FakeSegmenter())) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 415


def test_predict_rejects_oversized_upload() -> None:
    segmenter = FakeSegmenter()
    segmenter.config = InferenceConfig(
        model_path=Path("fake.onnx"),
        max_upload_bytes=4,
    )
    with TestClient(create_app(segmenter=segmenter)) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", b"12345", "image/png")},
        )

        assert response.status_code == 413
