"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from src.rml_vision_usecase.pipelines.train_model.nodes import (
    make_train_test_split,
    define_model_pipeline,
    define_model,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=make_train_test_split,
                inputs=[
                    "2014.data_with_salary",
                    "2015.data_with_salary",
                    "params:drop_features",
                    "params:new_features",
                ],
                outputs=[
                    "train_X",
                    "train_y",
                    "val_X",
                    "val_y",
                    "test_X",
                    "test_y",
                    "mlflow_train_dataset",
                    "mlflow_val_dataset",
                    "mlflow_test_dataset",
                ],
                name="make_train_test_split",
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
                func=train_model,
                inputs=[
                    "train_X",
                    "train_y",
                    "val_X",
                    "val_y",
                    "model_pipeline",
                    "params:model",
                    "mlflow_train_dataset",
                    "mlflow_val_dataset",
                ],
                outputs=["fitted_model", "validation_accuracy"],
                name="train_model",
            ),
            # Node(
            #     func=validate_model,
            #     inputs=["val_X", "val_y", "trained_model"],
            #     outputs="fitted_model",
            #     name="validate_model",
            # ),
        ]
    )


# base_train_pipeline = Pipeline(
#     [
#
#     ]
# )
#
#
# def create_logistic_regression_pipeline(**kwargs) -> Pipeline:
#     return Pipeline(
#         [base_train_pipeline],
#         namespace="linear_regression",
#         inputs={"train_X", "train_y"},
#     )
#
#
# def create_naive_bayes_pipeline(**kwargs) -> Pipeline:
#     return Pipeline(
#         [base_train_pipeline],
#         namespace="naive_bayes",
#         inputs={"train_X", "train_y"},
#     )
#
#
# def create_decision_tree_pipeline(**kwargs) -> Pipeline:
#     return Pipeline(
#         [base_train_pipeline],
#         namespace="decision_tree",
#         inputs={"train_X", "train_y"},
#     )
