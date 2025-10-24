"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from src.rml_vision_usecase.pipelines.train_model.nodes import (
    make_train_test_split,
    drop_columns,
    feature_engineering,
    define_model_pipeline,
    define_model,
    train,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=make_train_test_split,
                inputs=["2014.data_with_salary", "2015.data_with_salary"],
                outputs=["train_data", "test_data"],
                name="make_train_test_split",
            ),
            Node(
                func=drop_columns,
                inputs=["train_data", "test_data", "params:columns"],
                outputs=["train_X_1", "train_y", "test_X_1", "test_y"],
                name="drop_columns",
            ),
            Node(
                func=feature_engineering,
                inputs=["train_X_1", "test_X_1", "params:new_features"],
                outputs=["train_X", "test_X"],
                name="engineer_features",
            ),
            Node(
                func=define_model,
                inputs="params:model",
                outputs="model",
                name="define_model",
            ),
            Node(
                func=define_model_pipeline,
                inputs="model",
                outputs="model_pipeline",
                name="define_model_pipeline",
            ),
            Node(
                func=train,
                inputs=["train_X", "train_y", "model_pipeline"],
                outputs="fitted_model",
                name="train",
            ),
        ]
    )
