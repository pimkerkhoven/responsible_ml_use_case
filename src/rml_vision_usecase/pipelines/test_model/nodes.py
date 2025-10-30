"""
This is a boilerplate pipeline 'test_model'
generated using Kedro 1.0.0
"""

import math
import random

import mlflow
import mlflow.sklearn
from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve

from src.rml_vision_usecase.pipelines.train_model.evaluate_privacy import (
    evaluate_privacy,
)
from src.rml_vision_usecase.pipelines.train_model.make_radar_plot import (
    create_radar_plot,
)


def load_model(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    return model


def test_model(test_data, model):
    mlflow_test_data = mlflow.data.from_pandas(
        test_data, name="ACS_INCOME", targets="TARGET"
    )
    mlflow.log_input(mlflow_test_data, context="testing")

    test_y = test_data["TARGET"]
    test_X = test_data.drop(["TARGET"], axis=1)

    accuracy = model.score(test_X.copy(), test_y)

    if hasattr(model, "decision_function"):
        fpr, tpr, _ = roc_curve(test_y, model.decision_function(test_X.copy()))
        auc_score = roc_auc_score(test_y, model.decision_function(test_X.copy()))
    else:
        fpr, tpr, _ = roc_curve(test_y, model.predict_proba(test_X.copy())[:, 1])
        auc_score = roc_auc_score(test_y, model.predict_proba(test_X.copy())[:, 1])

    mlflow.log_metric("accuracy", accuracy, dataset=mlflow_test_data)
    mlflow.log_metric("auc_score", auc_score, dataset=mlflow_test_data)

    return (fpr, tpr), mlflow_test_data


def test_responsible_metrics_model(test_data, model, mlflow_test_data):
    test_y = test_data["TARGET"]

    if hasattr(model["clf"], "get_depth"):
        test_X = test_data.drop(["TARGET", "SEX"], axis=1)
    else:
        test_X = test_data.drop(["TARGET"], axis=1)

    sens_data = test_data["SEX"]

    y_pred = model.predict(test_X.copy())

    accuracy = accuracy_score(y_true=test_y, y_pred=y_pred)
    precision = precision_score(y_true=test_y, y_pred=y_pred)
    fairness = 1 - demographic_parity_difference(
        y_true=test_y, y_pred=y_pred, sensitive_features=sens_data
    )
    privacy = 1 - evaluate_privacy(test_X, sens_data, model)

    if hasattr(model["clf"], "get_depth"):
        tree_depth = model["clf"].get_depth()
        explainability = 1 - (tree_depth / 100)
    else:
        explainability = 0

    mlflow.log_metric("precision", precision, dataset=mlflow_test_data)
    mlflow.log_metric("fairness", fairness, dataset=mlflow_test_data)
    mlflow.log_metric("privacy", privacy, dataset=mlflow_test_data)
    mlflow.log_metric("explainability", explainability, dataset=mlflow_test_data)

    radar_plot = create_radar_plot(
        data=[precision, accuracy, privacy, fairness, explainability],
        labels=["Precision", "Accuracy", "Privacy", "Fairness", "Explainability"],
    )
    mlflow.log_figure(radar_plot, "radar_plot.png")


def _change_label_helper(chance=0):
    def change_label(label):
        if label and random.random() < chance:
            return False

        return label

    return change_label


def monitor_model(monitor_data, model):
    mlflow_monitor_data = mlflow.data.from_pandas(
        monitor_data, name="ACS_INCOME", targets="TARGET"
    )
    mlflow.log_input(mlflow_monitor_data, context="monitoring")

    if hasattr(model["clf"], "get_depth"):
        monitor_X = monitor_data.drop(["TARGET", "SEX"], axis=1)
    else:
        monitor_X = monitor_data.drop(["TARGET"], axis=1)

    sens_data = monitor_data["SEX"]
    monitor_y = monitor_data["TARGET"]

    prev = 0
    for i in range(0, len(monitor_X), math.floor(len(monitor_X) / 10)):
        if i == 0:
            continue

        X = monitor_X.iloc[prev:i].copy()
        y = monitor_y.iloc[prev:i].copy()
        sens = sens_data.iloc[prev:i].copy()

        # y.loc[sens == 1.0] = y[sens == 1.0].map(
        #     _change_label_helper(i / len(monitor_X))
        # )
        y = y.map(_change_label_helper(i / len(monitor_X)))

        y_pred = model.predict(X.copy())

        # print(y_pred)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y_true=y, y_pred=y_pred)
        fairness = 1 - demographic_parity_difference(
            y_true=y, y_pred=y_pred, sensitive_features=sens
        )
        privacy = 1 - evaluate_privacy(
            X=monitor_X.iloc[:i].copy(), sex=sens_data.iloc[:i].copy(), model=model
        )

        print(accuracy, precision, fairness, privacy)

        prev = i

    # y_pred = model.predict(monitor_X.copy())
