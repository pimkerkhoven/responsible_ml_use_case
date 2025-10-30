"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from src.rml_vision_usecase.pipelines.train_model.nodes import (
    define_model_pipeline,
    define_model,
    train_model,
    validate_model,
    define_responsible_model_pipeline,
    multi_objective_train_model,
    train_models_given_hyperparams,
    validate_models,
    train_fairness_ranker,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=define_model,
                inputs="params:model",
                outputs="model",
                name="define_model",
                tags=["traditional", "responsible"],
            ),
            Node(
                func=define_model_pipeline,
                inputs=[
                    "model",
                    "params:augment_features",
                    "params:drop_features",
                    "params:engineered_features",
                    "occ_to_sal",
                ],
                outputs="model_pipeline",
                name="define_model_pipeline",
                tags=["traditional", "responsible"],
            ),
            # Traditional training
            Node(
                func=train_model,
                inputs=[
                    "train_data",
                    "model_pipeline",
                    "params:model",
                ],
                outputs="fitted_model",
                name="train_model",
                tags="traditional",
            ),
            Node(
                func=validate_model,
                inputs=["validation_data", "fitted_model"],
                outputs="validation_accuracy",
                name="validate_model",
                tags="traditional",
            ),
            # Responsible training
            Node(
                func=train_fairness_ranker,
                inputs="train_data",
                outputs="fairness_ranker",
                name="train_fairness_ranker",
                tags="responsible_assist",
            ),
            Node(
                func=define_responsible_model_pipeline,
                inputs=["model_pipeline"],
                outputs="create_responsible_pipeline",
                name="define_responsible_model_pipeline",
                tags="responsible",
            ),
            Node(
                func=multi_objective_train_model,
                inputs=[
                    "train_data",
                    "create_responsible_pipeline",
                    "fairness_ranker",
                ],
                outputs="pareto_front_model_hyperparams",
                name="multi_objective_train_model",
                tags="responsible",
            ),
            Node(
                func=train_models_given_hyperparams,
                inputs=[
                    "train_data",
                    "pareto_front_model_hyperparams",
                    "create_responsible_pipeline",
                    "fairness_ranker",
                ],
                outputs="fitted_models",
                name="train_models_given_hyperparams",
                tags="responsible",
            ),
            Node(
                func=validate_models,
                inputs=[
                    "validation_data",
                    "fitted_models",
                    "params:model",
                ],
                outputs=None,
                name="validate_models",
                tags="responsible",
            ),
        ]
    )
