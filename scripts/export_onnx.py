"""Export the MLflow candidate model to ONNX and verify one inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow.db"
DEFAULT_MODEL_URI = "models:/factoryvision-segmentation@candidate"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "models" / "factoryvision-segmentation.onnx"
INPUT_SHAPE = (1, 3, 256, 640)


def export_and_verify(
    model_uri: str = DEFAULT_MODEL_URI,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> dict[str, float | str]:
    """Export the selected model and compare ONNX Runtime with PyTorch."""

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    torch.manual_seed(42)
    example_input = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        pytorch_output = model(example_input).cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example_input,),
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(
        ["logits"],
        {"images": example_input.numpy()},
    )[0]
    max_absolute_error = float(np.max(np.abs(pytorch_output - onnx_output)))
    mean_absolute_error = float(np.mean(np.abs(pytorch_output - onnx_output)))
    np.testing.assert_allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-3,
        atol=1e-5,
    )

    return {
        "model_uri": model_uri,
        "output_path": str(output_path),
        "input_shape": str(INPUT_SHAPE),
        "output_shape": str(tuple(onnx_output.shape)),
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    result = export_and_verify(
        model_uri=args.model_uri,
        output_path=args.output,
        tracking_uri=args.tracking_uri,
    )
    print("ONNX export and numerical verification succeeded.")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
