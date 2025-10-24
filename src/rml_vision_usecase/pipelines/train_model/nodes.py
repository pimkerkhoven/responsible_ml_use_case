"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def make_train_test_split(data_2014, data_2015):
    data = pd.concat([data_2014, data_2015], ignore_index=True)
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    return train, test


def drop_columns(train, test, columns):
    train_y = train["TARGET"]
    train_X = train.drop(["TARGET"], axis=1)
    train_X = train_X[columns]

    test_y = test["TARGET"]
    test_X = test.drop(["TARGET"], axis=1)
    test_X = test_X[columns]

    return train_X, train_y, test_X, test_y


def feature_engineering(train_X, test_X, features):
    if features is None:
        return train_X, test_X

    for data in [train_X, test_X]:
        if "AGE_WKHP" in features:
            data["AGE_WKHP"] = data["WKHP"] * data["AGEP"]
        if "AGE_WKHP_CAT" in features:
            data["AGE_WKHP_CAT"] = (data["WKHP"] * data["AGEP"]) > 1200

    return train_X, test_X

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


def define_model(parameters):
    if parameters["type"] == "logistic-regression":
        model = LogisticRegression(max_iter=10_000)

        # parameter_grid["clf__C"] = (0.1, 1)
    elif parameters["type"] == "decision-tree":
        model = DecisionTreeClassifier()

        # parameter_grid["clf__min_samples_leaf"] = [50, 60, 70]
    elif parameters["type"] == "naive-bayes":
        model = GaussianNB()

        model = Pipeline(
            steps=[
                ("convert_sparse", FunctionTransformer(lambda X: X.toarray())),
                ("model", model),
            ]
        )
    else:
        raise ValueError("Unknown model type")

    return model  # parameter_grid


def train(train_X, train_y, model):
    model.fit(train_X, train_y)

    return model
