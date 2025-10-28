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

mlflow.set_experiment("Initial Models")

with mlflow.start_run():
    for model_type in ["logistic_regression", "decision_tree", "naive_bayes"]:
        # Create and use the session
        with KedroSession.create(
            project_path=project_root, runtime_params={"model_type": model_type}
        ) as session:
            with mlflow.start_run(run_name=model_type, nested=True):
                session.run(
                    from_nodes=[
                        "2014.augment_data_with_salary",
                        "2015.augment_data_with_salary",
                        "2016.augment_data_with_salary",
                        "create_occ_to_sal",
                        "define_model",
                    ]
                )
