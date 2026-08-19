"""FastAPI application for FactoryVision real-time inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from .inference import InferenceConfig, InvalidImageError, OnnxSegmenter
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse


def create_app(
    segmenter: OnnxSegmenter | None = None,
    config: InferenceConfig | None = None,
) -> FastAPI:
    """Create the API and optionally inject a segmenter for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if segmenter is None:
            app.state.segmenter = OnnxSegmenter(config)
        else:
            app.state.segmenter = segmenter
        yield
        app.state.segmenter = None

    app = FastAPI(
        title="FactoryVision Inference API",
        version="0.1.0",
        description="Industrial surface-defect segmentation with ONNX Runtime.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report service liveness and model readiness."""

        loaded = getattr(request.app.state, "segmenter", None) is not None
        return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        """Return the loaded model's serving metadata."""

        segmenter = getattr(request.app.state, "segmenter", None)
        if segmenter is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The inference model is not loaded.",
            )
        return ModelInfoResponse(**segmenter.model_info())

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        file: UploadFile = File(..., description="An image to inspect."),
    ) -> PredictionResponse:
        """Return a defect mask, score, and bounding box for one image."""

        segmenter = getattr(request.app.state, "segmenter", None)
        if segmenter is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The inference model is not loaded.",
            )
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="The uploaded file must have an image/* content type.",
            )
        image_bytes = await file.read(segmenter.config.max_upload_bytes + 1)
        if len(image_bytes) > segmenter.config.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="The uploaded image is too large.",
            )
        try:
            prediction = segmenter.predict(image_bytes)
        except InvalidImageError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not process the image: {error}",
            ) from error
        return segmenter.prediction_response(prediction)

    return app


app = create_app()
