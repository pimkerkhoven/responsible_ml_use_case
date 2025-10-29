from itertools import chain, combinations, product
from pathlib import Path

import mlflow

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import configure_project
from kedro.utils import find_kedro_project


def powerset(iterable):
    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def input_to_run_name(af, df, ef, model_params):
    result = ""
    for k, v in model_params.items():
        result += f" {k}={v}"

    if len(af) + len(ef) > 0:
        result += f" with: {', '.join(af + ef)}"

    if len(df) > 0:
        result += f" without: {', '.join(df)}"

    if result == "":
        return "Base model"

    return result


# Get project root
current_dir = Path(__file__).resolve().parent
project_root = find_kedro_project(current_dir)
bootstrap_project(Path(project_root))

mlflow.set_experiment("Improved Models")

augment_features = ["MEAN_SALARY", "HOUR_RATE"]
drop_features = ["COW", "POBP"]
engineered_features = ["AGE_WKHP", "AGE_WKHP_CAT"]
all_model_params = {
    "logistic_regression": {"C": [0.1, 1]},
    "decision_tree": {"min_samples_leaf": [1, 50, 60, 70]},
    "naive_bayes": {},
}

with mlflow.start_run():
    for model_type in ["logistic_regression", "decision_tree", "naive_bayes"]:
        with mlflow.start_run(run_name=model_type, nested=True):
            model_params = all_model_params[model_type]

            model_param_keys = list(model_params.keys())
            model_param_value_combos = list(product(*model_params.values()))

            best_validation_accuracy = 0
            best_af = None
            best_ef = None
            best_df = None
            best_model_params = None

            for af in powerset(augment_features):
                for df in powerset(drop_features):
                    for ef in powerset(engineered_features):
                        for param_combo in model_param_value_combos:
                            run_model_params = {
                                model_param_keys[i]: v
                                for i, v in enumerate(param_combo)
                            }

                            run_name = input_to_run_name(af, df, ef, run_model_params)

                            # Create and use the session
                            with KedroSession.create(
                                project_path=project_root,
                                runtime_params={
                                    "model_type": model_type,
                                    "augment_features": list(af),
                                    "engineered_features": list(ef),
                                    "drop_features": list(df),
                                    "model_params": run_model_params,
                                },
                            ) as session:
                                with mlflow.start_run(nested=True, run_name=run_name):
                                    mlflow.log_param("Model type", model_type)
                                    mlflow.log_param("Augmented features", list(af))
                                    mlflow.log_param("Engineered features", list(ef))
                                    mlflow.log_param("Dropped features", list(df))
                                    for k, v in run_model_params.items():
                                        mlflow.log_param(k, v)

                                    res = session.run(pipeline_name="train_model")

                                    validation_accuracy = res[
                                        "validation_accuracy"
                                    ].load()
                                    if validation_accuracy > best_validation_accuracy:
                                        best_validation_accuracy = validation_accuracy
                                        best_af = af
                                        best_df = df
                                        best_ef = ef
                                        best_model_params = run_model_params

            mlflow.log_param("Best model type", model_type)
            mlflow.log_param("Best augmented features", list(best_af))
            mlflow.log_param("Best engineered features", list(best_ef))
            mlflow.log_param("Best dropped features", list(best_df))
            mlflow.log_metric("Best validation accuracy", best_validation_accuracy)
            for k, v in best_model_params.items():
                mlflow.log_param(f"Best {k}", v)
