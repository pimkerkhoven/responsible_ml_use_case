"""
This is a boilerplate pipeline 'test_model'
generated using Kedro 1.0.0
"""

import mlflow
import mlflow.sklearn


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

    test_accuracy = model.score(test_X, test_y)
    mlflow.log_metric("test_accuracy", test_accuracy, dataset=mlflow_test_data)

    return test_accuracy
