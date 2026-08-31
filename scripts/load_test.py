"""Run a small concurrent load test against the FactoryVision prediction API."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import mimetypes
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


DEFAULT_URL = "http://127.0.0.1:8000/predict"
DEFAULT_IMAGE = Path("assets/dataset/non_defect_sample.jpg")
DEFAULT_OUTPUT = Path("artifacts/load-test/report.json")


@dataclass(frozen=True)
class RequestResult:
    """Outcome and client-observed duration of one prediction request."""

    latency_ms: float
    status_code: int | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the request received a successful HTTP response."""

        return self.status_code is not None and 200 <= self.status_code < 300


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a nearest-rank percentile without interpolation surprises."""

    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def summarize_results(
    results: Sequence[RequestResult],
    *,
    duration_seconds: float,
    target_url: str,
    image_path: Path,
    concurrency: int,
) -> dict[str, object]:
    """Build a JSON-serializable report from completed request results."""

    total_requests = len(results)
    successful_requests = sum(result.succeeded for result in results)
    failed_requests = total_requests - successful_requests
    latencies = [result.latency_ms for result in results]
    status_counts = Counter(
        str(result.status_code) if result.status_code is not None else "connection_error"
        for result in results
    )
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    safe_duration = max(duration_seconds, 1e-9)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_url": target_url,
        "image": str(image_path),
        "total_requests": total_requests,
        "concurrency": concurrency,
        "duration_seconds": round(duration_seconds, 4),
        "throughput_requests_per_second": round(total_requests / safe_duration, 4),
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "error_rate_percent": round((failed_requests / total_requests) * 100, 4)
        if total_requests
        else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "latency_ms": {
            "min": round(min(latencies), 4) if latencies else 0.0,
            "mean": round(mean_latency, 4),
            "p50": round(percentile(latencies, 0.50), 4),
            "p95": round(percentile(latencies, 0.95), 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "errors": [result.error for result in results if result.error][:5],
    }


def _multipart_payload(
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> tuple[bytes, str]:
    """Create one multipart/form-data body for the API's UploadFile input."""

    boundary = f"FactoryVisionLoadTest{uuid.uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    return prefix + image_bytes + suffix, f"multipart/form-data; boundary={boundary}"


def send_prediction(
    target_url: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    timeout_seconds: float,
) -> RequestResult:
    """Send one request and convert transport/HTTP failures into a result."""

    started_at = perf_counter()
    body, multipart_type = _multipart_payload(image_bytes, filename, content_type)
    request = Request(
        target_url,
        data=body,
        headers={"Content-Type": multipart_type},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response.read()
            status_code = response.status
            error = None if 200 <= status_code < 300 else f"HTTP {status_code}"
    except HTTPError as error:
        status_code = error.code
        error = f"HTTP {error.code}: {error.reason}"
    except (OSError, TimeoutError, URLError) as error:
        status_code = None
        error = str(error)
    return RequestResult(
        latency_ms=(perf_counter() - started_at) * 1000.0,
        status_code=status_code,
        error=error,
    )


def run_load_test(
    target_url: str,
    image_path: Path,
    request_count: int,
    concurrency: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """Run concurrent prediction requests and return their summary report."""

    image_bytes = image_path.read_bytes()
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    results: list[RequestResult] = []
    started_at = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                send_prediction,
                target_url,
                image_bytes,
                image_path.name,
                content_type,
                timeout_seconds,
            )
            for _ in range(request_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    duration_seconds = perf_counter() - started_at
    return summarize_results(
        results,
        duration_seconds=duration_seconds,
        target_url=target_url,
        image_path=image_path,
        concurrency=concurrency,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Prediction endpoint URL.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--requests", type=int, default=20, dest="request_count")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0, dest="timeout_seconds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.request_count <= 0 or arguments.concurrency <= 0:
        parser.error("--requests and --concurrency must be positive")
    if arguments.timeout_seconds <= 0:
        parser.error("--timeout must be positive")
    if not arguments.image.is_file():
        parser.error(f"image does not exist: {arguments.image}")
    return arguments


def main() -> int:
    """Run the command-line load test and write its JSON report."""

    arguments = _arguments()
    report = run_load_test(
        arguments.url,
        arguments.image,
        arguments.request_count,
        arguments.concurrency,
        arguments.timeout_seconds,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    latency = report["latency_ms"]
    print(f"Saved load-test report to {arguments.output}")
    print(f"Requests: {report['total_requests']} | errors: {report['failed_requests']}")
    print(f"Throughput: {report['throughput_requests_per_second']} requests/s")
    print(f"Latency p50: {latency['p50']} ms | p95: {latency['p95']} ms")
    print(f"Error rate: {report['error_rate_percent']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
