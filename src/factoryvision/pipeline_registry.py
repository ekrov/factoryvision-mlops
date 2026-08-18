"""Register the Kedro pipelines exposed by FactoryVision."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Return named pipelines and the default pipeline."""

    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    return pipelines
