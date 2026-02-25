"""
This is a boilerplate pipeline 'test_model'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from src.rml_vision_usecase.pipelines.test_model.nodes import (
    load_model,
    test_model,
    test_responsible_metrics_model,
    monitor_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=load_model,
                inputs=["params:model_name", "params:model_version"],
                outputs="loaded_model",
                name="load_model",
                tags=["initial", "improved", "responsible"],
            ),
            Node(
                func=test_model,
                inputs=["test_data", "loaded_model"],
                outputs=["roc_curve", "mlflow_test_data"],
                name="test_model",
                tags=["initial", "improved", "responsible"],
            ),
            Node(
                func=test_responsible_metrics_model,
                inputs=["test_data", "loaded_model", "mlflow_test_data"],
                outputs="metrics",
                name="test_responsible_metrics_model",
                tags=["responsible"],
            ),
            Node(
                func=monitor_model,
                inputs=["2016.data", "loaded_model", "metrics"],
                outputs=None,
                name="monitor_model",
                tags=["responsible"],
            ),
        ]
    )
