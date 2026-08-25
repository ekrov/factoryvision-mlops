"""Pydantic response schemas for the FactoryVision API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Inclusive pixel coordinates around the predicted defect region."""

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)


class HealthResponse(BaseModel):
    """Service liveness and model readiness state."""

    status: str
    model_loaded: bool
    storage_ready: bool


class ModelInfoResponse(BaseModel):
    """Metadata describing the loaded inference model."""

    model_name: str
    model_alias: str
    runtime: str
    input_shape: list[int | str]
    output_shape: list[int | str]
    threshold: float


class PredictionResponse(BaseModel):
    """Segmentation and image-level defect results."""

    model_name: str
    model_alias: str
    runtime: str
    has_defect: bool
    defect_probability: float = Field(ge=0.0, le=1.0)
    defect_area_fraction: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox | None
    mask_base64: str
    mask_media_type: str
    mask_height: int = Field(gt=0)
    mask_width: int = Field(gt=0)
    original_image_height: int = Field(gt=0)
    original_image_width: int = Field(gt=0)
