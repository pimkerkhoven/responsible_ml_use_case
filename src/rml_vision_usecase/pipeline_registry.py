"""Project pipelines."""

from __future__ import annotations

# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from src.rml_vision_usecase.pipelines.prepare_data import (
    create_pipeline as create_prepare_data_pipeline,
)
from src.rml_vision_usecase.pipelines.prepare_data.pipeline import (
    create_2014_pipeline,
    create_2015_pipeline,
    create_2016_pipeline,
)
from src.rml_vision_usecase.pipelines.test_model import (
    create_pipeline as create_test_model_pipeline,
)
from src.rml_vision_usecase.pipelines.train_model import (
    create_pipeline as create_train_model_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()

    # pipelines["__default__"] = sum(pipelines.values())
    pipelines = dict()

    pipelines["prepare_data"] = create_prepare_data_pipeline()
    pipelines["prepare_data_2014"] = create_2014_pipeline()
    pipelines["prepare_data_2015"] = create_2015_pipeline()
    pipelines["prepare_data_2016"] = create_2016_pipeline()

    pipelines["prepare_all_data"] = (
        pipelines["prepare_data_2014"]
        + pipelines["prepare_data_2015"]
        + pipelines["prepare_data_2016"]
        + pipelines["prepare_data"]
    )

    pipelines["train_model"] = create_train_model_pipeline()
    pipelines["test_model"] = create_test_model_pipeline()

    return pipelines
