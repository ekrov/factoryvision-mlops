"""Build a reviewable end-to-end FactoryVision demo GIF."""

from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont, ImageOps

from factoryvision.api.inference import InferenceConfig, OnnxSegmenter
from factoryvision.api.main import create_app


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "demo"
WIDTH, HEIGHT = 1280, 720
BACKGROUND = "#0f172a"
PANEL = "#1e293b"
TEXT = "#f8fafc"
MUTED = "#cbd5e1"
GREEN = "#86efac"
ORANGE = "#fdba74"


class DemoStore:
    """Minimal in-memory store for exercising the real API route."""

    def __init__(self) -> None:
        self.successes = 0

    def save_success(self, **_: object) -> None:
        self.successes += 1

    def save_failure(self, **_: object) -> None:
        return None


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a Windows font, with Pillow's default as a portable fallback."""

    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def write_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int = 24,
    fill: str = TEXT,
    bold: bool = False,
) -> None:
    """Draw one line using the demo's shared typography."""

    draw.text(xy, value, font=load_font(size, bold), fill=fill)


def panel(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    """Draw a rounded panel behind one demo section."""

    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=18, fill=PANEL)


def header(title: str, subtitle: str) -> Image.Image:
    """Create a consistent slide canvas."""

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    write_text(draw, (48, 34), title, 34, bold=True)
    write_text(draw, (50, 82), subtitle, 18, MUTED)
    write_text(draw, (48, 678), "FactoryVision | local evidence walkthrough", 16, MUTED)
    return canvas


def contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit an image inside a box without stretching it."""

    return ImageOps.contain(image.convert("RGB"), size)


def centered_paste(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    """Fit and center an image inside a canvas rectangle."""

    left, top, right, bottom = box
    fitted = contain(image, (right - left, bottom - top))
    x = left + (right - left - fitted.width) // 2
    y = top + (bottom - top - fitted.height) // 2
    canvas.paste(fitted, (x, y))


def run_api_demo() -> tuple[dict[str, object], Image.Image]:
    """Call the real FastAPI prediction route and build its overlay."""

    image_path = ROOT / "assets" / "dataset" / "defect_sample.png"
    segmenter = OnnxSegmenter(InferenceConfig())
    store = DemoStore()
    app = create_app(segmenter=segmenter, prediction_store=store)
    image_bytes = image_path.read_bytes()
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, image_bytes, "image/png")},
        )
    response.raise_for_status()
    payload = response.json()
    mask_bytes = base64.b64decode(payload["mask_base64"])
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    original = Image.open(image_path).convert("RGB")
    target_size = (payload["mask_width"], payload["mask_height"])
    letterboxed = Image.new("RGB", target_size)
    fitted = contain(original, target_size)
    letterboxed.paste(
        fitted,
        ((letterboxed.width - fitted.width) // 2, (letterboxed.height - fitted.height) // 2),
    )
    red = Image.new("RGB", letterboxed.size, (239, 68, 68))
    highlighted = Image.composite(red, letterboxed, mask)
    overlay = Image.blend(letterboxed, highlighted, 0.45)
    return payload, overlay


def title_slide() -> Image.Image:
    """Create the opening architecture slide."""

    canvas = header("FactoryVision", "End-to-end industrial surface-defect segmentation")
    draw = ImageDraw.Draw(canvas)
    write_text(draw, (80, 170), "Upload", 24, bold=True)
    write_text(draw, (300, 170), "->", 28, ORANGE, bold=True)
    write_text(draw, (380, 170), "FastAPI", 24, bold=True)
    write_text(draw, (590, 170), "->", 28, ORANGE, bold=True)
    write_text(draw, (670, 170), "ONNX Runtime", 24, bold=True)
    write_text(draw, (960, 170), "->", 28, ORANGE, bold=True)
    write_text(draw, (1040, 170), "Prediction", 24, bold=True)
    panel(canvas, (80, 265, 1200, 540))
    write_text(draw, (120, 310), "The demo follows one inspection image through:", 25, bold=True)
    for index, label in enumerate(
        [
            "1. API prediction and mask generation",
            "2. PostgreSQL-compatible prediction persistence",
            "3. Prometheus metrics and Grafana observability",
            "4. MLflow experiment comparison and model registry",
        ]
    ):
        write_text(draw, (145, 365 + index * 38), label, 21, GREEN if index == 0 else TEXT)
    return canvas


def api_slide(payload: dict[str, object]) -> Image.Image:
    """Create the API request and response slide."""

    canvas = header(
        "1 | Upload and predict",
        "The GIF calls the real FastAPI /predict route with an in-memory demo store",
    )
    panel(canvas, (48, 135, 420, 625))
    centered_paste(
        canvas,
        Image.open(ROOT / "assets" / "dataset" / "defect_sample.png"),
        (72, 165, 396, 580),
    )
    draw = ImageDraw.Draw(canvas)
    write_text(draw, (72, 590), "defect_sample.png", 17, MUTED)
    panel(canvas, (470, 135, 1232, 625))
    write_text(draw, (510, 170), "POST /predict", 26, GREEN, bold=True)
    write_text(draw, (510, 220), "multipart file upload", 19, MUTED)
    response_lines = [
        "status: 200",
        f"has_defect: {str(payload['has_defect']).lower()}",
        f"defect_probability: {payload['defect_probability']:.6f}",
        f"defect_area_fraction: {payload['defect_area_fraction']:.6f}",
        f"mask: {payload['mask_height']} x {payload['mask_width']}",
        f"bounding_box: {payload['bounding_box']}",
    ]
    for index, line in enumerate(response_lines):
        write_text(draw, (540, 285 + index * 42), line, 22, TEXT if index else GREEN)
    write_text(draw, (510, 555), "The response contains the mask, score, and derived box.", 18, MUTED)
    return canvas


def prediction_slide(payload: dict[str, object], overlay: Image.Image) -> Image.Image:
    """Create the visual segmentation result slide."""

    canvas = header(
        "2 | Inspect the segmentation result",
        "The predicted mask is returned as a base64-encoded PNG",
    )
    panel(canvas, (48, 135, 790, 625))
    centered_paste(canvas, overlay, (72, 165, 766, 580))
    draw = ImageDraw.Draw(canvas)
    write_text(draw, (72, 590), "Predicted defect overlay", 20, GREEN, bold=True)
    panel(canvas, (830, 135, 1232, 625))
    write_text(draw, (860, 175), "Post-processing", 24, bold=True)
    lines = [
        "sigmoid(logits)",
        "threshold >= 0.5",
        "binary mask",
        "area fraction",
        "bounding box",
        "",
        "This example is a",
        "real model result,",
        "not a mock image.",
    ]
    for index, line in enumerate(lines):
        write_text(draw, (865, 230 + index * 34), line, 19, ORANGE if index == 1 else TEXT)
    return canvas


def grafana_slide() -> Image.Image:
    """Create the monitoring slide from the captured dashboard."""

    canvas = header(
        "3 | Observe the service",
        "Prometheus scrapes /metrics and Grafana turns time series into panels",
    )
    dashboard = Image.open(ROOT / "assets" / "screenshots" / "grafana-dashboard.png")
    panel(canvas, (48, 135, 1232, 625))
    centered_paste(canvas, dashboard, (65, 150, 1215, 610))
    draw = ImageDraw.Draw(canvas)
    write_text(
        draw,
        (80, 615),
        "Throughput | HTTP errors | latency | defect rate | model version",
        18,
        GREEN,
        bold=True,
    )
    return canvas


def mlflow_slide() -> Image.Image:
    """Create an MLflow comparison slide from the real mini-study CSV."""

    canvas = header(
        "4 | Compare and register the model",
        "MLflow stores parameters, metrics, checkpoints, and the selected alias",
    )
    draw = ImageDraw.Draw(canvas)
    panel(canvas, (48, 135, 1232, 625))
    write_text(draw, (82, 172), "Experiment: factoryvision-mini-study", 25, GREEN, bold=True)
    write_text(draw, (82, 214), "Registered model: factoryvision-segmentation@candidate", 21, ORANGE)
    headers = ["Configuration", "IoU", "Dice", "Defect F1"]
    x_positions = [90, 600, 790, 1000]
    for x, label in zip(x_positions, headers, strict=True):
        write_text(draw, (x, 280), label, 20, MUTED, bold=True)
    with (ROOT / "artifacts" / "mini-study" / "results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row_index, row in enumerate(rows[:3]):
        y = 335 + row_index * 60
        values = [
            row["name"],
            f"{float(row['final_iou']):.4f}",
            f"{float(row['final_dice']):.4f}",
            f"{float(row['final_defect_f1']):.4f}",
        ]
        for x, value in zip(x_positions, values, strict=True):
            write_text(draw, (x, y), value, 21, TEXT if row["name"] != "baseline" else GREEN)
    write_text(draw, (90, 550), "Winner: baseline, selected by final validation IoU.", 21, GREEN, bold=True)
    write_text(draw, (90, 585), "The two-epoch study is a candidate comparison, not a final accuracy claim.", 17, MUTED)
    return canvas


def closing_slide() -> Image.Image:
    """Create the closing reproducibility slide."""

    canvas = header(
        "5 | Reproduce the workflow",
        "The same path can be run locally with the documented commands",
    )
    draw = ImageDraw.Draw(canvas)
    panel(canvas, (80, 160, 1200, 575))
    commands = [
        ".venv\\Scripts\\python.exe scripts\\export_onnx.py",
        "docker compose up --build",
        "POST http://localhost:8000/predict",
        "open Grafana and MLflow",
    ]
    for index, command in enumerate(commands):
        write_text(draw, (145, 220 + index * 58), command, 24, GREEN if index == 2 else TEXT)
    write_text(draw, (145, 485), "Evidence and limitations are documented in README.md, MODEL_CARD.md, and DATA_CARD.md.", 19, MUTED)
    write_text(draw, (145, 520), "The full narration and live-demo checklist are in docs/demo.md.", 19, MUTED)
    return canvas


def main() -> None:
    """Build the demo GIF, storyboard preview, overlay, and response evidence."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload, overlay = run_api_demo()
    frames = [
        title_slide(),
        api_slide(payload),
        prediction_slide(payload, overlay),
        grafana_slide(),
        mlflow_slide(),
        closing_slide(),
    ]
    gif_path = OUT_DIR / "factoryvision-demo.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=[12000, 24000, 30000, 30000, 30000, 24000],
        loop=0,
        optimize=True,
    )
    preview = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    thumbnails = [ImageOps.contain(frame, (400, 225)) for frame in frames]
    for index, thumbnail in enumerate(thumbnails):
        x = 35 + (index % 3) * 415
        y = 110 + (index // 3) * 280
        preview.paste(thumbnail, (x, y))
    preview.save(OUT_DIR / "factoryvision-demo-storyboard.png")
    overlay.save(OUT_DIR / "api-prediction-overlay.png")
    (OUT_DIR / "api-response.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {gif_path}")
    print(f"Saved {OUT_DIR / 'factoryvision-demo-storyboard.png'}")
    print(f"Saved {OUT_DIR / 'api-prediction-overlay.png'}")
    print(f"Saved {OUT_DIR / 'api-response.json'}")


if __name__ == "__main__":
    main()
