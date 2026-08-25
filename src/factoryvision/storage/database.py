"""Database engine and schema initialization for prediction storage."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .models import Base


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://factoryvision:factoryvision@"
    "localhost:5432/factoryvision"
)


def database_url_from_environment() -> str:
    """Read the database URL, defaulting to local PostgreSQL development settings."""

    return os.getenv("FACTORYVISION_DATABASE_URL", DEFAULT_DATABASE_URL)


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL or an injected test database."""

    url = database_url or database_url_from_environment()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


def initialize_database(engine: Engine) -> None:
    """Create the prediction table if it does not exist yet."""

    Base.metadata.create_all(engine)
