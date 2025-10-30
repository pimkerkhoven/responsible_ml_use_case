from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.utils import find_kedro_project

# Get project root
current_dir = Path(__file__).resolve().parent
project_root = find_kedro_project(current_dir)
bootstrap_project(Path(project_root))

mlflow.set_experiment("Test Models")

runs = {
    "initial": [
        ("Logistic regression", "initial_logistic_regression"),
        ("Decision tree", "initial_decision_tree"),
        ("Naive bayes", "initial_naive_bayes"),
    ],
    "improved": [
        ("Logistic regression", "improved_logistic_regression"),
        ("Decision tree", "improved_decision_tree"),
        ("Naive bayes", "improved_naive_bayes"),
    ],
}

with mlflow.start_run():
    for run_name, _ in runs.items():
        with mlflow.start_run(run_name=run_name, nested=True):
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.set_ylim(0, 1.01)
            ax.set_xlim(-0.01, 1)
            ax.set_title("ROC Curves")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")

            for model_name, model_path in runs[run_name]:
                # Create and use the session
                with KedroSession.create(
                    project_path=project_root,
                    runtime_params={
                        "model_name": model_path,
                        "model_version": "latest",
                    },
                ) as session:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        res = session.run(pipeline_name="test_model")
                        (fpr, tpr) = res["roc_curve"].load()

                        ax.plot(fpr, tpr, label=model_name.title())

                plt.legend(loc=4)
                plt.grid()
                mlflow.log_figure(fig, f"roc_curves_{run_name}.png")
                plt.close()
