"""FastAPI application for FactoryVision real-time inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError

from .inference import InferenceConfig, InvalidImageError, OnnxSegmenter
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse
from factoryvision.storage.database import create_database_engine, initialize_database
from factoryvision.storage.repository import PredictionRepository, image_id_from_bytes
from factoryvision.monitoring.metrics import (
    HTTP_ERRORS,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    INFERENCE_LATENCY,
    MODEL_INFO,
    PREDICTIONS,
    metrics_payload,
)


def create_app(
    segmenter: OnnxSegmenter | None = None,
    config: InferenceConfig | None = None,
    prediction_store: PredictionRepository | None = None,
) -> FastAPI:
    """Create the API and optionally inject dependencies for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if segmenter is None:
            app.state.segmenter = OnnxSegmenter(config)
        else:
            app.state.segmenter = segmenter
        MODEL_INFO.labels(
            app.state.segmenter.config.model_name,
            app.state.segmenter.config.model_alias,
            app.state.segmenter.runtime_name,
        ).set(1)
        if prediction_store is None:
            engine = create_database_engine()
            initialize_database(engine)
            app.state.prediction_store = PredictionRepository.from_engine(engine)
            app.state.database_engine = engine
        else:
            app.state.prediction_store = prediction_store
            app.state.database_engine = None
        try:
            yield
        finally:
            if app.state.database_engine is not None:
                app.state.database_engine.dispose()
        app.state.segmenter = None
        app.state.prediction_store = None

    app = FastAPI(
        title="FactoryVision Inference API",
        version="0.1.0",
        description="Industrial surface-defect segmentation with ONNX Runtime.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def collect_http_metrics(request: Request, call_next):
        def request_path() -> str:
            route = request.scope.get("route")
            return getattr(route, "path", request.url.path)

        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            path = request_path()
            HTTP_REQUESTS.labels(request.method, path, "500").inc()
            HTTP_ERRORS.labels(request.method, path, "500").inc()
            HTTP_LATENCY.labels(request.method, path).observe(
                time.perf_counter() - started_at
            )
            raise
        path = request_path()
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        if response.status_code >= 400:
            HTTP_ERRORS.labels(request.method, path, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, path).observe(
            time.perf_counter() - started_at
        )
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Expose API metrics for Prometheus scraping."""

        payload, content_type = metrics_payload()
        return Response(content=payload, media_type=content_type)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report service liveness and model readiness."""

        loaded = getattr(request.app.state, "segmenter", None) is not None
        storage_ready = getattr(request.app.state, "prediction_store", None) is not None
        return HealthResponse(
            status="ok" if loaded and storage_ready else "degraded",
            model_loaded=loaded,
            storage_ready=storage_ready,
        )

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
        prediction_store = getattr(request.app.state, "prediction_store", None)
        if prediction_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction storage is not ready.",
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
        image_id = image_id_from_bytes(image_bytes)
        started_at = time.perf_counter()
        inference_started_at = time.perf_counter()
        inference_labels = (
            segmenter.config.model_name,
            segmenter.config.model_alias,
        )

        def record_inference_duration() -> None:
            INFERENCE_LATENCY.labels(*inference_labels).observe(
                time.perf_counter() - inference_started_at
            )

        def record_prediction_outcome(outcome: str) -> None:
            PREDICTIONS.labels(
                outcome,
                segmenter.config.model_name,
                segmenter.config.model_alias,
            ).inc()

        def save_failure(error_message: str) -> None:
            """Best-effort persistence for failed inference attempts."""

            try:
                prediction_store.save_failure(
                    image_id=image_id,
                    model_name=segmenter.config.model_name,
                    model_alias=segmenter.config.model_alias,
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    error_message=error_message,
                )
            except SQLAlchemyError:
                # Preserve the original inference error if persistence also fails.
                pass

        try:
            prediction = segmenter.predict(image_bytes)
        except InvalidImageError as error:
            record_inference_duration()
            record_prediction_outcome("error")
            save_failure(str(error))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except (RuntimeError, ValueError) as error:
            record_inference_duration()
            record_prediction_outcome("error")
            save_failure(str(error))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not process the image: {error}",
            ) from error
        except Exception:
            record_inference_duration()
            record_prediction_outcome("error")
            raise
        record_inference_duration()
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        try:
            prediction_store.save_success(
                image_id=image_id,
                model_name=segmenter.config.model_name,
                model_alias=segmenter.config.model_alias,
                prediction=prediction,
                latency_ms=latency_ms,
            )
        except SQLAlchemyError as error:
            record_prediction_outcome("error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Prediction completed but could not be stored: {error}",
            ) from error
        record_prediction_outcome("defect" if prediction.has_defect else "no_defect")
        return segmenter.prediction_response(prediction)

    return app


app = create_app()
