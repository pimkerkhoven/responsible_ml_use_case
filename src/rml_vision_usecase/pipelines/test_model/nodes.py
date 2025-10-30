"""
This is a boilerplate pipeline 'test_model'
generated using Kedro 1.0.0
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, roc_curve


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

    accuracy = model.score(test_X, test_y)

    if hasattr(model, "decision_function"):
        fpr, tpr, _ = roc_curve(test_y, model.decision_function(test_X))
        auc_score = roc_auc_score(test_y, model.decision_function(test_X))
    else:
        fpr, tpr, _ = roc_curve(test_y, model.predict_proba(test_X)[:, 1])
        auc_score = roc_auc_score(test_y, model.predict_proba(test_X)[:, 1])

    mlflow.log_metric("accuracy", accuracy, dataset=mlflow_test_data)
    mlflow.log_metric("auc_score", auc_score, dataset=mlflow_test_data)

    return (fpr, tpr)
