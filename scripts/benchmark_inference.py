"""Benchmark PyTorch and ONNX Runtime inference for the FactoryVision model."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import onnxruntime as ort
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow.db"
DEFAULT_MODEL_URI = "models:/factoryvision-segmentation@candidate"
DEFAULT_ONNX_PATH = REPO_ROOT / "artifacts" / "models" / "factoryvision-segmentation.onnx"
DEFAULT_RESULTS_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "inference.json"
INPUT_SHAPE = (1, 3, 256, 640)


def _artifact_size_bytes(path: Path) -> int:
    """Return the size of a file or all files in a model artifact directory."""

    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _measure(
    inference: Callable[[], np.ndarray],
    warmup_runs: int,
    measured_runs: int,
) -> tuple[list[float], np.ndarray]:
    """Warm up an inference function and measure elapsed milliseconds."""

    last_output: np.ndarray | None = None
    for _ in range(warmup_runs):
        last_output = inference()

    latencies_ms = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        last_output = inference()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    if last_output is None:
        raise RuntimeError("Inference did not produce an output.")
    return latencies_ms, last_output


def _summary(latencies_ms: list[float]) -> dict[str, float]:
    """Calculate latency and throughput statistics."""

    mean_latency_ms = float(np.mean(latencies_ms))
    return {
        "mean_latency_ms": mean_latency_ms,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_images_per_second": 1000.0 / mean_latency_ms,
    }


def _downloaded_model_size(model_uri: str) -> int:
    """Measure the complete MLflow PyTorch model artifact bundle."""

    with tempfile.TemporaryDirectory(prefix="factoryvision-mlflow-") as temp_dir:
        downloaded_path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=temp_dir,
            )
        )
        return _artifact_size_bytes(downloaded_path)


def benchmark(
    model_uri: str = DEFAULT_MODEL_URI,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    warmup_runs: int = 10,
    measured_runs: int = 30,
    threads: int = 1,
) -> dict[str, object]:
    """Run a controlled CPU comparison and save its results as JSON."""

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model does not exist: {onnx_path}. "
            "Run scripts/export_onnx.py first."
        )
    if warmup_runs < 1 or measured_runs < 1 or threads < 1:
        raise ValueError("warmup_runs, measured_runs, and threads must be positive.")

    mlflow.set_tracking_uri(tracking_uri)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    example_input = torch.randn(INPUT_SHAPE, dtype=torch.float32)

    pytorch_model = mlflow.pytorch.load_model(model_uri)
    pytorch_model.eval()

    def run_pytorch() -> np.ndarray:
        with torch.no_grad():
            return pytorch_model(example_input).cpu().numpy()

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    def run_onnx() -> np.ndarray:
        return session.run(["logits"], {"images": example_input.numpy()})[0]

    pytorch_latencies, pytorch_output = _measure(
        run_pytorch,
        warmup_runs,
        measured_runs,
    )
    onnx_latencies, onnx_output = _measure(
        run_onnx,
        warmup_runs,
        measured_runs,
    )
    np.testing.assert_allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-3,
        atol=1e-5,
    )

    pytorch_stats = _summary(pytorch_latencies)
    onnx_stats = _summary(onnx_latencies)
    result: dict[str, object] = {
        "model_uri": model_uri,
        "onnx_path": str(onnx_path),
        "device": "cpu",
        "threads": threads,
        "input_shape": list(INPUT_SHAPE),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "pytorch_mlflow_bundle_size_mb": round(
            _downloaded_model_size(model_uri) / (1024 * 1024),
            3,
        ),
        "onnx_file_size_mb": round(
            onnx_path.stat().st_size / (1024 * 1024),
            3,
        ),
        "pytorch": pytorch_stats,
        "onnxruntime": onnx_stats,
        "max_absolute_output_error": float(
            np.max(np.abs(pytorch_output - onnx_output))
        ),
        "mean_absolute_output_error": float(
            np.mean(np.abs(pytorch_output - onnx_output))
        ),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    result = benchmark(
        model_uri=args.model_uri,
        onnx_path=args.onnx,
        results_path=args.output,
        tracking_uri=args.tracking_uri,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        threads=args.threads,
    )
    print("Inference benchmark succeeded.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
