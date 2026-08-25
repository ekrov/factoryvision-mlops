"""SQLAlchemy models for persisted FactoryVision predictions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for FactoryVision database tables."""


class PredictionRecord(Base):
    """One inference event and its operational metadata."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(255))
    model_alias: Mapped[str] = mapped_column(String(100))
    defect_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    defect_area_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    x_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    y_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    x_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    y_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
