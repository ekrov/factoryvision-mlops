"""Tests for the PySpark daily prediction aggregation."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import sys

import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402

from spark.daily_prediction_statistics import derive_daily_statistics  # noqa: E402


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[2]")
        .appName("FactoryVisionAnalyticsTests")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    yield session
    session.stop()


def test_daily_statistics_classifies_defects_and_errors(spark: SparkSession) -> None:
    logs = spark.createDataFrame(
        [
            (
                datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc),
                "factoryvision-segmentation",
                "candidate",
                0.90,
                10.0,
                "success",
            ),
            (
                datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc),
                "factoryvision-segmentation",
                "candidate",
                0.10,
                20.0,
                "success",
            ),
            (
                datetime(2026, 8, 20, 11, 0, tzinfo=timezone.utc),
                "factoryvision-segmentation",
                "candidate",
                None,
                5.0,
                "error",
            ),
        ],
        [
            "created_at",
            "model_name",
            "model_alias",
            "defect_probability",
            "latency_ms",
            "status",
        ],
    )

    rows = {
        row.defect_status: row.asDict()
        for row in derive_daily_statistics(logs).collect()
    }

    assert set(rows) == {"defect", "no_defect", "error"}
    assert rows["defect"]["prediction_count"] == 1
    assert rows["no_defect"]["prediction_count"] == 1
    assert rows["error"]["prediction_count"] == 1
    assert rows["defect"]["successful_predictions"] == 2
    assert rows["defect"]["defect_predictions"] == 1
    assert rows["defect"]["error_predictions"] == 1
    assert rows["defect"]["defect_rate"] == pytest.approx(0.5)
    assert rows["defect"]["error_rate"] == pytest.approx(1 / 3)
