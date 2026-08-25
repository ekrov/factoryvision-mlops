"""Scheduled FactoryVision batch-inference workflow."""

from __future__ import annotations

from pathlib import Path

from airflow.sdk import dag, get_current_context, task
from pendulum import datetime

from factoryvision.batch import (
    batch_config_from_environment,
    discover_new_images,
    persist_prediction_artifact,
    run_inference_to_file,
    validate_image_paths,
    write_summary_artifact,
)
from factoryvision.api.inference import InferenceConfig
from factoryvision.storage.database import create_database_engine
from factoryvision.storage.repository import PredictionRepository


@dag(
    dag_id="factoryvision_batch_inference",
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["factoryvision", "computer-vision", "batch-inference"],
)
def factoryvision_batch_inference():
    """Discover, inspect, persist, and summarize new FactoryVision images."""

    @task(retries=2)
    def discover_images() -> list[str]:
        config = batch_config_from_environment()
        engine = create_database_engine(config.database_url)
        repository = PredictionRepository.from_engine(engine)
        try:
            return discover_new_images(
                config.image_dir,
                repository,
                max_images=config.max_images,
            )
        finally:
            engine.dispose()

    @task(retries=1)
    def validate_images(image_paths: list[str]) -> list[str]:
        return validate_image_paths(image_paths)

    @task(retries=1)
    def run_batch_inference(image_paths: list[str]) -> str:
        context = get_current_context()
        config = batch_config_from_environment()
        run_id = str(context["run_id"]).replace(":", "_").replace("/", "_")
        prediction_path = config.artifact_dir / run_id / "predictions.json"
        return run_inference_to_file(
            image_paths,
            prediction_path,
            config=InferenceConfig.from_environment(),
        )

    @task(retries=2)
    def persist_results(prediction_path: str) -> dict[str, object]:
        return persist_prediction_artifact(
            Path(prediction_path),
            config=batch_config_from_environment(),
        )

    @task(retries=1)
    def generate_summary(
        prediction_path: str,
        persisted_summary: dict[str, object],
    ) -> str:
        config = batch_config_from_environment()
        return write_summary_artifact(
            Path(prediction_path),
            persisted_summary,
            config.artifact_dir,
        )

    discovered = discover_images()
    validated = validate_images(discovered)
    predictions = run_batch_inference(validated)
    persisted = persist_results(predictions)
    generate_summary(predictions, persisted)


factoryvision_batch_inference()
