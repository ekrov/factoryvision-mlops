"""Kedro pipeline connecting the FactoryVision baseline stages."""

from kedro.pipeline import Node, Pipeline

from . import nodes


def create_pipeline() -> Pipeline:
    """Create the ingestion, preprocessing, training, and evaluation pipeline."""

    return Pipeline(
        [
            Node(
                func=nodes.ingest_datasets,
                inputs=["params:dataset", "params:augmentation"],
                outputs=["train_dataset", "validation_dataset"],
                name="ingest_kolektor_sdd2_node",
            ),
            Node(
                func=nodes.prepare_dataloaders,
                inputs=[
                    "train_dataset",
                    "validation_dataset",
                    "params:dataset",
                    "params:training",
                ],
                outputs=["train_loader", "validation_loader"],
                name="prepare_dataloaders_node",
            ),
            Node(
                func=nodes.train_model,
                inputs=[
                    "train_loader",
                    "validation_loader",
                    "params:dataset",
                    "params:training",
                    "params:model",
                    "params:augmentation",
                    "params:tracking",
                ],
                outputs=[
                    "trained_model",
                    "training_history",
                    "checkpoint_path",
                    "mlflow_run_id",
                ],
                name="train_unet_node",
            ),
            Node(
                func=nodes.evaluate_model,
                inputs=[
                    "trained_model",
                    "train_dataset",
                    "validation_dataset",
                    "validation_loader",
                    "params:dataset",
                    "params:training",
                    "params:evaluation",
                    "params:tracking",
                    "mlflow_run_id",
                ],
                outputs=["evaluation_metrics", "validation_preview_path"],
                name="evaluate_unet_node",
            ),
        ]
    )
