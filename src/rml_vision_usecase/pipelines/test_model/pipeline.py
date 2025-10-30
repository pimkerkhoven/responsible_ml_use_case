"""
This is a boilerplate pipeline 'test_model'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from src.rml_vision_usecase.pipelines.test_model.nodes import load_model, test_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=load_model,
                inputs=["params:model_name", "params:model_version"],
                outputs="loaded_model",
                name="load_model",
            ),
            Node(
                func=test_model,
                inputs=["test_data", "loaded_model"],
                outputs="roc_curve",
                name="test_model",
            ),
        ]
    )
