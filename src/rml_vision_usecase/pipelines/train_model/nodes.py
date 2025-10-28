"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 1.0.0
"""

import numpy as np
import pandas as pd


import mlflow


from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def _engineer_features(X, new_features):
    if new_features is None:
        return X

    if "AGE_WKHP" in new_features:
        X["AGE_WKHP"] = X["WKHP"] * X["AGEP"]
    if "AGE_WKHP_CAT" in new_features:
        X["AGE_WKHP_CAT"] = (X["WKHP"] * X["AGEP"]) > 1200

    return X


def _log_data_in_mlflow(test_X, test_y, train_X, train_y, val_X, val_y):
    train_data = pd.concat([train_X, train_y], axis=1)
    val_data = pd.concat([val_X, val_y], axis=1)
    test_data = pd.concat([test_X, test_y], axis=1)

    train_dataset = mlflow.data.from_pandas(
        train_data, name="ACS_INCOME", targets="TARGET"
    )
    val_dataset = mlflow.data.from_pandas(val_data, name="ACS_INCOME", targets="TARGET")
    test_dataset = mlflow.data.from_pandas(
        test_data, name="ACS_INCOME", targets="TARGET"
    )

    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(val_dataset, context="validation")
    mlflow.log_input(test_dataset, context="testing")

    return train_dataset, val_dataset, test_dataset


def make_train_test_split(data_2014, data_2015, drop_features, new_features):
    if drop_features is None:
        drop_features = []

    data = pd.concat([data_2014, data_2015], ignore_index=True)

    y = data["TARGET"]
    X = data.drop(drop_features + ["TARGET"], axis=1)

    X = _engineer_features(X, new_features)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_X, val_X, train_y, val_y = train_test_split(
        train_X, train_y, test_size=0.2, random_state=42
    )

    mlflow_train_dataset, mlflow_val_dataset, mlflow_test_dataset = _log_data_in_mlflow(
        test_X,
        test_y,
        train_X,
        train_y,
        val_X,
        val_y,
    )

    return (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
        mlflow_train_dataset,
        mlflow_val_dataset,
        mlflow_test_dataset,
    )


def _to_category(x):
    return x.astype("category")


def define_model_pipeline(model):
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
    ]

    return Pipeline(
        steps=[
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
                                        OneHotEncoder(
                                            handle_unknown="infrequent_if_exist"
                                        ),
                                    ),
                                ]
                            ),
                            make_column_selector("|".join(categorical_features)),
                        ),
                    ]
                ),
            ),
            ("clf", model),
        ]
    )


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
            model = DecisionTreeClassifier(min_samples_leaf=parameters["params"]["min_samples_leaf"])
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


def train_model(
    train_X,
    train_y,
    val_X,
    val_y,
    model,
    parameters,
    mlflow_train_dataset,
    mlflow_val_dataset,
):
    model.fit(train_X, train_y)

    mlflow.log_metric(
        "training_accuracy", model.score(train_X, train_y), dataset=mlflow_train_dataset
    )
    validation_accuracy = model.score(val_X, val_y)
    mlflow.log_metric(
        "validation_accuracy", validation_accuracy, dataset=mlflow_val_dataset
    )
    signature = mlflow.models.infer_signature(train_X, model.predict(train_X))

    mlflow.sklearn.log_model(
        sk_model=model,
        name=parameters["type"],
        signature=signature,
        input_example=train_X,
        # registered_model_name="tracking-quickstart",
    )

    return model, validation_accuracy


# def validate_model(val_X, val_y, model):
#     model.score(val_X, val_y)
#
#     return model
