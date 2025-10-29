from pathlib import Path

import mlflow

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import configure_project
from kedro.utils import find_kedro_project

# Get project root
current_dir = Path(__file__).resolve().parent
project_root = find_kedro_project(current_dir)
bootstrap_project(Path(project_root))

mlflow.set_experiment("Test Initial Models")

model_names = [
    "initial_logistic_regression",
    "initial_decision_tree",
    "initial_naive_bayes",
    "improved_logistic_regression",
    "improved_decision_tree",
    "improved_naive_bayes",
]

with mlflow.start_run():
    for model_name in model_names:
        # Create and use the session
        with KedroSession.create(
            project_path=project_root,
            runtime_params={"model_name": model_name, "model_version": "latest"},
        ) as session:
            with mlflow.start_run(run_name=model_name, nested=True):
                session.run(pipeline_name="test_model")
