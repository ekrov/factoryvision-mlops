"""Repository operations for persisted inference results."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from factoryvision.api.inference import Prediction

from .models import PredictionRecord


def image_id_from_bytes(image_bytes: bytes) -> str:
    """Create a stable identifier without storing the uploaded image itself."""

    return hashlib.sha256(image_bytes).hexdigest()


class PredictionRepository:
    """Persist inference success and failure records through SQLAlchemy."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    @classmethod
    def from_engine(cls, engine: Any) -> "PredictionRepository":
        """Build a repository backed by a SQLAlchemy engine."""

        return cls(sessionmaker(bind=engine, expire_on_commit=False))

    def save_success(
        self,
        image_id: str,
        model_name: str,
        model_alias: str,
        prediction: Prediction,
        latency_ms: float,
    ) -> PredictionRecord:
        """Save a successful prediction and return its database record."""

        return self.save_success_metadata(
            image_id=image_id,
            model_name=model_name,
            model_alias=model_alias,
            defect_probability=prediction.defect_probability,
            defect_area_fraction=prediction.defect_area_fraction,
            bounding_box=prediction.bounding_box,
            latency_ms=latency_ms,
        )

    def save_success_metadata(
        self,
        image_id: str,
        model_name: str,
        model_alias: str,
        defect_probability: float,
        defect_area_fraction: float,
        bounding_box: Any,
        latency_ms: float,
    ) -> PredictionRecord:
        """Save success fields without requiring the response mask in memory."""

        record = PredictionRecord(
            image_id=image_id,
            created_at=datetime.now(timezone.utc),
            model_name=model_name,
            model_alias=model_alias,
            defect_probability=defect_probability,
            defect_area_fraction=defect_area_fraction,
            x_min=bounding_box.x_min if bounding_box else None,
            y_min=bounding_box.y_min if bounding_box else None,
            x_max=bounding_box.x_max if bounding_box else None,
            y_max=bounding_box.y_max if bounding_box else None,
            latency_ms=latency_ms,
            status="success",
        )
        return self._save(record)

    def save_failure(
        self,
        image_id: str,
        model_name: str,
        model_alias: str,
        latency_ms: float,
        error_message: str,
    ) -> PredictionRecord:
        """Save an inference failure without pretending a prediction exists."""

        record = PredictionRecord(
            image_id=image_id,
            created_at=datetime.now(timezone.utc),
            model_name=model_name,
            model_alias=model_alias,
            defect_probability=None,
            defect_area_fraction=None,
            latency_ms=latency_ms,
            status="error",
            error_message=error_message,
        )
        return self._save(record)

    def _save(self, record: PredictionRecord) -> PredictionRecord:
        with self.session_factory.begin() as session:
            session.add(record)
        return record

    def latest_for_image(self, image_id: str) -> PredictionRecord | None:
        """Return the latest stored event for one image identifier."""

        statement = (
            select(PredictionRecord)
            .where(PredictionRecord.image_id == image_id)
            .order_by(PredictionRecord.created_at.desc())
            .limit(1)
        )
        with self.session_factory() as session:
            return session.scalar(statement)
