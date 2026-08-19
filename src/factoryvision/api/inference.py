"""ONNX Runtime inference and image post-processing for FactoryVision."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .schemas import BoundingBox, PredictionResponse


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "models" / "factoryvision-segmentation.onnx"
DEFAULT_MODEL_NAME = "factoryvision-segmentation"
DEFAULT_MODEL_ALIAS = "candidate"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_UPLOAD_BYTES = 10_000_000
TARGET_HEIGHT = 256
TARGET_WIDTH = 640


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


@dataclass(frozen=True)
class InferenceConfig:
    """Serving settings that must remain aligned with training/export."""

    model_path: Path = DEFAULT_MODEL_PATH
    model_name: str = DEFAULT_MODEL_NAME
    model_alias: str = DEFAULT_MODEL_ALIAS
    threshold: float = DEFAULT_THRESHOLD
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    image_height: int = TARGET_HEIGHT
    image_width: int = TARGET_WIDTH

    @classmethod
    def from_environment(cls) -> "InferenceConfig":
        """Build serving settings from environment overrides and defaults."""

        raw_model_path = Path(
            os.getenv("FACTORYVISION_ONNX_MODEL", str(DEFAULT_MODEL_PATH))
        )
        model_path = (
            raw_model_path
            if raw_model_path.is_absolute()
            else REPO_ROOT / raw_model_path
        )
        threshold = float(
            os.getenv("FACTORYVISION_THRESHOLD", str(DEFAULT_THRESHOLD))
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("FACTORYVISION_THRESHOLD must be between 0 and 1")
        return cls(
            model_path=model_path,
            model_name=os.getenv("FACTORYVISION_MODEL_NAME", DEFAULT_MODEL_NAME),
            model_alias=os.getenv("FACTORYVISION_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
            threshold=threshold,
            max_upload_bytes=int(
                os.getenv(
                    "FACTORYVISION_MAX_UPLOAD_BYTES",
                    str(DEFAULT_MAX_UPLOAD_BYTES),
                )
            ),
        )


@dataclass(frozen=True)
class Prediction:
    """Internal prediction representation before API serialization."""

    has_defect: bool
    defect_probability: float
    defect_area_fraction: float
    bounding_box: BoundingBox | None
    mask_base64: str
    original_image_height: int
    original_image_width: int
    mask_height: int
    mask_width: int


def _letterbox_image(image: np.ndarray, config: InferenceConfig) -> np.ndarray:
    """Apply the same aspect-preserving resize and padding as the dataset loader."""

    original_height, original_width = image.shape[:2]
    scale = min(
        config.image_height / original_height,
        config.image_width / original_width,
    )
    resized_height = min(
        config.image_height,
        max(1, round(original_height * scale)),
    )
    resized_width = min(
        config.image_width,
        max(1, round(original_width * scale)),
    )
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    top = (config.image_height - resized_height) // 2
    left = (config.image_width - resized_width) // 2
    bottom = config.image_height - resized_height - top
    right = config.image_width - resized_width - left
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REFLECT_101,
    )


def preprocess_image(
    image_bytes: bytes,
    config: InferenceConfig,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode and transform an upload into an ONNX input tensor."""

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("The uploaded bytes are not a readable image.")
    original_height, original_width = image.shape[:2]
    image = _letterbox_image(image, config)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), (original_height, original_width)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities without overflow for extreme values."""

    clipped = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _bounding_box(binary_mask: np.ndarray) -> BoundingBox | None:
    """Find the smallest predicted-defect rectangle, or None if empty."""

    y_coordinates, x_coordinates = np.where(binary_mask)
    if len(x_coordinates) == 0:
        return None
    return BoundingBox(
        x_min=int(x_coordinates.min()),
        y_min=int(y_coordinates.min()),
        x_max=int(x_coordinates.max()),
        y_max=int(y_coordinates.max()),
    )


def _encode_mask(binary_mask: np.ndarray) -> str:
    """Encode the binary mask as a compact base64 PNG for JSON transport."""

    mask_image = (binary_mask.astype(np.uint8) * 255)
    success, encoded = cv2.imencode(".png", mask_image)
    if not success:
        raise RuntimeError("Could not encode the predicted mask as PNG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class OnnxSegmenter:
    """Load one ONNX Runtime session and serve binary defect segmentation."""

    runtime_name = "onnxruntime"

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig.from_environment()
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.config.model_path}. "
                "Run scripts/export_onnx.py first."
            )
        self.session = ort.InferenceSession(
            str(self.config.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_bytes: bytes) -> Prediction:
        """Run preprocessing, ONNX inference, thresholding, and post-processing."""

        input_tensor, original_size = preprocess_image(image_bytes, self.config)
        logits = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )[0]
        probabilities = _sigmoid(logits[0, 0])
        binary_mask = probabilities >= self.config.threshold
        return Prediction(
            has_defect=bool(binary_mask.any()),
            defect_probability=float(probabilities.max()),
            defect_area_fraction=float(binary_mask.mean()),
            bounding_box=_bounding_box(binary_mask),
            mask_base64=_encode_mask(binary_mask),
            original_image_height=original_size[0],
            original_image_width=original_size[1],
            mask_height=int(binary_mask.shape[0]),
            mask_width=int(binary_mask.shape[1]),
        )

    def model_info(self) -> dict[str, object]:
        """Return model metadata for the API endpoint."""

        return {
            "model_name": self.config.model_name,
            "model_alias": self.config.model_alias,
            "runtime": self.runtime_name,
            "input_shape": [1, 3, self.config.image_height, self.config.image_width],
            "output_shape": [1, 1, self.config.image_height, self.config.image_width],
            "threshold": self.config.threshold,
        }

    def prediction_response(self, prediction: Prediction) -> PredictionResponse:
        """Convert an internal prediction into the public response schema."""

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
