from pathlib import Path

import mlflow
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.utils import find_kedro_project

# Get project root
current_dir = Path(__file__).resolve().parent
project_root = find_kedro_project(current_dir)
bootstrap_project(Path(project_root))

mlflow.set_experiment("Responsible Models")


with KedroSession.create(
    project_path=project_root, runtime_params={"model_type": "decision_tree"}
) as session:
    with mlflow.start_run():
        session.run(pipeline_name="train_model", tags=["responsible"])

# kedro run --pipeline=train_model --tags=responsible --params=model_type=decision_tree
