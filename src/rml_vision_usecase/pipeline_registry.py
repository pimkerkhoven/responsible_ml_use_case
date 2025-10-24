"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from src.rml_vision_usecase.pipelines.prepare_data.pipeline import (
    create_2014_pipeline,
    create_2015_pipeline,
    create_2016_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["prepare_data_2014"] = create_2014_pipeline()
    pipelines["prepare_data_2015"] = create_2015_pipeline()
    pipelines["prepare_data_2016"] = create_2016_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
