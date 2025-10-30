"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 1.0.0
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from platypus import NSGAII, Problem, Real, nondominated
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.rml_vision_usecase.pipelines.train_model.anonymize_data import anonymizeData
from src.rml_vision_usecase.pipelines.train_model.evaluate_privacy import (
    evaluate_privacy,
)
from src.rml_vision_usecase.pipelines.train_model.make_radar_plot import (
    create_radar_plot,
)
from src.rml_vision_usecase.pipelines.train_model.massage_data import massage_data


class AugmentFeatures(BaseEstimator):
    def __init__(
        self,
        occ_to_sal,
        features=None,
    ):
        self.features = features
        self.occ_to_sal = occ_to_sal

    def transform(self, X, y=None):
        if "MEAN_SALARY" in self.features:
            X["MEAN_SALARY"] = (
                X["OCCP"]
                .astype(int)
                .map(
                    lambda x: self.occ_to_sal[x][0] if x in self.occ_to_sal else np.nan
                )
            )

        if "HOUR_RATE" in self.features:
            X["HOUR_RATE"] = (
                X["OCCP"]
                .astype(int)
                .map(
                    lambda x: self.occ_to_sal[x][1] if x in self.occ_to_sal else np.nan
                )
                * X["WKHP"]
            )

        return X

    def fit(self, X, y=None):
        return self


class EngineerFeatures(BaseEstimator):
    def __init__(self, features=None):
        self.features = features

    def transform(self, X, y=None):
        if "AGE_WKHP" in self.features:
            X["AGE_WKHP"] = X["WKHP"] * X["AGEP"]
            if "AGE_WKHP_CAT" in self.features:
                X["AGE_WKHP_CAT"] = (X["WKHP"] * X["AGEP"]) > 1200

        return X

    def fit(self, X, y=None):
        return self


class DropFeatures(BaseEstimator):
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


def _to_category(x):
    return x.astype("category")


def define_model_pipeline(
    model, augment_features, drop_features, engineered_features, occ_to_sal
):
    numerical_features = [
        "AGEP",
        "WKHP",
        "AGE_WKHP_NUM",
        "MEAN_SALARY",
        "HOUR_RATE",
    ]
    categorical_features = [
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP",
        "SEX",
        "RAC1P",
        "AGE_WKHP_CAT",
        "AGE_GEN",
        "WKH_GEN",
    ]

    steps = []

    if augment_features is not None and len(augment_features) > 0:
        steps.append(
            ("augment_features", AugmentFeatures(occ_to_sal, augment_features))
        )

    if drop_features is not None and len(drop_features) > 0:
        steps.append(("drop_features", DropFeatures(drop_features)))

    if engineered_features is not None and len(engineered_features) > 0:
        steps.append(("engineered_features", EngineerFeatures(engineered_features)))

    steps.append(
        (
            "transform_columns",
            ColumnTransformer(
                transformers=[
                    (
                        "numerical",
                        Pipeline(
                            steps=[
                                (
                                    "impute",
                                    SimpleImputer(
                                        missing_values=np.nan, strategy="mean"
                                    ),
                                ),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        make_column_selector("|".join(numerical_features)),
                    ),
                    (
                        "categorical",
                        Pipeline(
                            steps=[
                                (
                                    "change_type",
                                    FunctionTransformer(_to_category),
                                ),
                                (
                                    "encoder",
                                    OneHotEncoder(handle_unknown="infrequent_if_exist"),
                                ),
                            ]
                        ),
                        make_column_selector("|".join(categorical_features)),
                    ),
                ]
            ),
        )
    )
    steps.append(("clf", model))

    return Pipeline(steps=steps)


def _df_to_array(df):
    return df.toarray()


def define_model(parameters):
    if parameters["type"] == "logistic_regression":
        if parameters["params"] is not None:
            model = LogisticRegression(max_iter=10_000, C=parameters["params"]["C"])
        else:
            model = LogisticRegression(max_iter=10_000)
    elif parameters["type"] == "decision_tree":
        if parameters["params"] is not None:
            model = DecisionTreeClassifier(
                min_samples_leaf=parameters["params"]["min_samples_leaf"]
            )
        else:
            model = DecisionTreeClassifier()
    elif parameters["type"] == "naive_bayes":
        model = GaussianNB()

        model = Pipeline(
            steps=[
                ("convert_sparse", FunctionTransformer(_df_to_array)),
                ("model", model),
            ]
        )
    else:
        raise ValueError("Unknown model type")

    return model  # parameter_grid


def train_model(train_data, model, parameters):
    mlflow_train_data = mlflow.data.from_pandas(
        train_data, name="ACS_INCOME", targets="TARGET"
    )
    mlflow.log_input(mlflow_train_data, context="training")

    train_y = train_data["TARGET"]
    train_X = train_data.drop(["TARGET"], axis=1)

    model.fit(train_X, train_y)

    mlflow.log_metric(
        "training_accuracy", model.score(train_X, train_y), dataset=mlflow_train_data
    )
    signature = mlflow.models.infer_signature(
        train_X.head(5), model.predict(train_X.head(5))
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        name=parameters["type"],
        signature=signature,
        input_example=train_X.head(5),
        # registered_model_name="tracking-quickstart",
    )

    return model


def _find_transform_columns_step_index(steps):
    for i, step in enumerate(steps):
        if step[0] == "transform_columns":
            return i

    raise ValueError("No transform columns found")


def define_responsible_model_pipeline(o_model_pipeline):
    def create_responsible_pipeline(transformation, min_samples_leaf):
        model_pipeline = deepcopy(o_model_pipeline)

        transform_columns_step_index = _find_transform_columns_step_index(
            model_pipeline.steps
        )

        model_pipeline.steps.insert(
            transform_columns_step_index,
            ("anonymize", anonymizeData(transformation=transformation)),
        )

        model_pipeline["clf"].min_samples_leaf = min_samples_leaf

        return model_pipeline

    return create_responsible_pipeline


def train_fairness_ranker(train_data):
    return LogisticRegression(max_iter=10_000).fit(
        train_data.drop(["TARGET", "SEX"], axis=1), train_data["TARGET"]
    )


SAMPLE_SIZE = 0.005
VAL_SIZE = 0.25


def multi_objective_train_model(data, create_model_pipeline, fairness_ranker):
    privacy_X = data.sample(frac=0.025, axis=0)
    privacy_sex = privacy_X["SEX"]
    privacy_X = privacy_X.drop(["TARGET", "SEX"], axis=1)

    def _train_tiny_model(variables):
        generalization_scheme = tuple(round(v) for v in (variables[:6]))
        fairness_level = variables[6]
        min_samples_leaf = variables[7]

        print(
            f"Execute train for - (Generalizer):{generalization_scheme}, f:{fairness_level}, d:{min_samples_leaf}"
        )

        # Sample subset of all training data to generate train and validation data
        train_data = data.sample(frac=SAMPLE_SIZE, axis=0)
        y = train_data["TARGET"]
        sensitive_feature = train_data["SEX"]
        X = train_data.drop(["TARGET", "SEX"], axis=1)
        X_train, X_val, y_train, y_val, sens_train, sens_val = train_test_split(
            X, y, sensitive_feature, test_size=VAL_SIZE
        )

        # Preprocess data
        y_massaged = massage_data(
            X_train,
            y_train,
            sensitive_feature=sens_train,
            f_perc=fairness_level,
            ranker=fairness_ranker,
        )

        model = create_model_pipeline(
            transformation=generalization_scheme, min_samples_leaf=min_samples_leaf
        )
        model.fit(X_train, y_massaged)

        # Create model pipeline for validation and generating scores

        # Calculate scores
        tree_depth = model["clf"].get_depth()

        y_val_pred = model.predict(X_val)

        model_accuracy = accuracy_score(y_true=y_val, y_pred=y_val_pred)
        model_precision = precision_score(y_true=y_val, y_pred=y_val_pred)
        model_fairness = demographic_parity_difference(
            y_true=y_val, y_pred=y_val_pred, sensitive_features=sens_val
        )
        model_privacy = evaluate_privacy(privacy_X, privacy_sex, model)

        # We want to minimize all the targets
        print(
            f"Training results - depth: {tree_depth}, precision: {1 - model_precision}, accuracy: {1 - model_accuracy},fairness: {model_fairness}, privacy: {model_privacy}"
        )
        return [
            tree_depth,
            1 - model_precision,
            1 - model_accuracy,
            model_fairness,
            model_privacy,
        ]

    MAX_FUNC_EVAL = 10_000
    POPULATION_SIZE = 50

    # Define optimization problem
    opt_problem = Problem(8, 5)
    opt_problem.types[:] = [
        # Hierarchies
        Real(0, 3),
        Real(0, 5),
        Real(0, 3),
        Real(0, 2),
        Real(0, 4),
        Real(0, 2),
        # Fairness
        Real(0, 1),
        # Explainability
        Real(0, 1),
    ]
    opt_problem.function = _train_tiny_model
    algorithm = NSGAII(opt_problem, population_size=POPULATION_SIZE)

    # Run optimization process
    algorithm.run(
        MAX_FUNC_EVAL, callback=lambda a: print(f"ALG STEP: {a.nfe}/{MAX_FUNC_EVAL}")
    )

    # Store parameters used to train the models on the Pareto-front
    nondominated_solutions = nondominated(algorithm.result)

    print(nondominated_solutions)

    pareto_variables = []
    for sol in nondominated_solutions:
        variables = sol.variables
        g = tuple(round(v) for v in (variables[:6]))
        f = variables[6]
        d = variables[7]

        pareto_variables.append((g, f, d))

    print(pareto_variables)

    return pareto_variables


def validate_model(validation_data, model):
    mlflow_val_data = mlflow.data.from_pandas(
        validation_data, name="ACS_INCOME", targets="TARGET"
    )
    mlflow.log_input(mlflow_val_data, context="validation")

    val_y = validation_data["TARGET"]
    val_X = validation_data.drop(["TARGET"], axis=1)

    validation_accuracy = model.score(val_X, val_y)
    mlflow.log_metric(
        "validation_accuracy", validation_accuracy, dataset=mlflow_val_data
    )

    return validation_accuracy


def train_models_given_hyperparams(
    train_data, hyperparams, create_model_pipeline, fairness_ranker
):
    mlflow_train_data = mlflow.data.from_pandas(
        train_data, name="ACS_INCOME", targets="TARGET"
    )
    mlflow.log_input(mlflow_train_data, context="training")

    train_y = train_data["TARGET"]
    train_X = train_data.drop(["TARGET", "SEX"], axis=1)
    sens_train = train_data["SEX"]

    models = []
    for i, (transformation, fairness_level, min_samples_leaf) in enumerate(hyperparams):
        print(f"{i + 1}/{len(hyperparams)}")
        print(
            f"START - Transformation:{transformation}, "
            f"fairness_level:{fairness_level:.2f}, "
            f"min_samples_leaf:{min_samples_leaf:.2f}"
        )

        y_massaged = massage_data(
            train_X,
            train_y,
            sensitive_feature=sens_train,
            f_perc=fairness_level,
            ranker=fairness_ranker,
        )

        model = create_model_pipeline(
            transformation=transformation, min_samples_leaf=min_samples_leaf
        )
        model.fit(train_X, y_massaged)

        models.append(model)

    return models


def validate_models(validation_data, models, model_params):
    mlflow_val_data = mlflow.data.from_pandas(
        validation_data, name="ACS_INCOME", targets="TARGET"
    )
    mlflow.log_input(mlflow_val_data, context="validation")
    val_y = validation_data["TARGET"]
    val_X = validation_data.drop(["TARGET", "SEX"], axis=1)
    sens_data = validation_data["SEX"]

    for i, model in enumerate(models):
        with mlflow.start_run(nested=True):
            tree_depth = model["clf"].get_depth()

            y_pred = model.predict(val_X)

            validation_accuracy = accuracy_score(y_true=val_y, y_pred=y_pred)
            validation_precision = precision_score(y_true=val_y, y_pred=y_pred)
            validation_fairness = 1 - demographic_parity_difference(
                y_true=val_y, y_pred=y_pred, sensitive_features=sens_data
            )
            validation_privacy = 1 - evaluate_privacy(val_X, sens_data, model)
            validation_explainability = 1 - (tree_depth / 100)

            # validation_accuracy = model.score(val_X, val_y)
            # f"Training results - depth: {tree_depth}, precision: {1 - model_precision}, accuracy: {1 - model_accuracy},fairness: {model_fairness}, privacy: {model_privacy}"

            mlflow.log_metric(
                "validation_accuracy", validation_accuracy, dataset=mlflow_val_data
            )
            mlflow.log_metric(
                "validation_precision", validation_precision, dataset=mlflow_val_data
            )
            mlflow.log_metric(
                "validation_fairness", validation_fairness, dataset=mlflow_val_data
            )
            mlflow.log_metric(
                "validation_privacy", validation_privacy, dataset=mlflow_val_data
            )
            mlflow.log_metric(
                "validation_explainability",
                validation_explainability,
                dataset=mlflow_val_data,
            )

            radar_plot = create_radar_plot(
                data=[
                    validation_precision,
                    validation_accuracy,
                    validation_privacy,
                    validation_fairness,
                    validation_explainability,
                ],
                labels=[
                    "Precision",
                    "Accuracy",
                    "Privacy",
                    "Fairness",
                    "Explainability",
                ],
                color_index=i,
            )
            mlflow.log_figure(radar_plot, f"radar_plot_{i}.png")
            plt.close()

            signature = mlflow.models.infer_signature(
                val_X.head(5), model.predict(val_X.head(5))
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_params["type"],
                signature=signature,
                input_example=val_X.head(5),
            )
