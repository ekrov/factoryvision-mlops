"""Prometheus metrics used by the inference API."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


HTTP_REQUESTS = Counter(
    "factoryvision_http_requests_total",
    "Total number of HTTP requests handled by the API.",
    ("method", "path", "status"),
)
HTTP_LATENCY = Histogram(
    "factoryvision_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "path"),
)
PREDICTIONS = Counter(
    "factoryvision_predictions_total",
    "Total number of completed predictions by outcome.",
    ("outcome", "model_name", "model_alias"),
)
MODEL_INFO = Gauge(
    "factoryvision_model_info",
    "Metadata for the model currently loaded by the API.",
    ("model_name", "model_alias", "runtime"),
)


def metrics_payload() -> tuple[bytes, str]:
    """Return the current metrics in Prometheus exposition format."""

    return generate_latest(), CONTENT_TYPE_LATEST
