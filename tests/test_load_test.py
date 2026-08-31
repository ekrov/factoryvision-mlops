"""Tests for the small FactoryVision load-test report calculations."""

from pathlib import Path

import pytest

from scripts.load_test import RequestResult, percentile, summarize_results


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([40.0, 10.0, 30.0, 20.0], 0.50) == 20.0
    assert percentile([40.0, 10.0, 30.0, 20.0], 0.95) == 40.0


def test_percentile_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 1.1)


def test_summary_reports_latency_throughput_and_errors() -> None:
    results = [
        RequestResult(10.0, 200),
        RequestResult(20.0, 200),
        RequestResult(30.0, 422, "HTTP 422: invalid image"),
        RequestResult(40.0, None, "connection refused"),
    ]

    report = summarize_results(
        results,
        duration_seconds=2.0,
        target_url="http://localhost:8000/predict",
        image_path=Path("sample.png"),
        concurrency=2,
    )

    assert report["successful_requests"] == 2
    assert report["failed_requests"] == 2
    assert report["error_rate_percent"] == 50.0
    assert report["throughput_requests_per_second"] == 2.0
    assert report["status_counts"] == {"200": 2, "422": 1, "connection_error": 1}
    assert report["latency_ms"] == {
        "min": 10.0,
        "mean": 25.0,
        "p50": 20.0,
        "p95": 40.0,
        "max": 40.0,
    }
