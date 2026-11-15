"""Compare two forecasting methods and track their data version and results."""

import hashlib
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = Path("data/monthly_counts.csv")
CUTOFF = 30


def main():
    dataset_bytes = DATA.read_bytes()
    lock = yaml.safe_load(Path("dvc.lock").read_text(encoding="utf-8"))
    expected_md5 = lock["stages"]["prepare"]["outs"][0]["md5"]
    if hashlib.md5(dataset_bytes, usedforsecurity=False).hexdigest() != expected_md5:
        raise ValueError("Dataset does not match dvc.lock. Run dvc repro before training.")

    data = pd.read_csv(DATA).sort_values(["area", "period"])
    data["previous_count"] = data.groupby("area")["reported_count"].shift()
    data = data.dropna(subset=["previous_count"])
    train = data[data["period"] < CUTOFF]
    test = data[data["period"] >= CUTOFF]
    target = "reported_count"
    features = ["area", "period", "month", "previous_count"]

    preprocess = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), ["area", "month"])],
        remainder="passthrough",
    )
    model = make_pipeline(preprocess, Ridge(alpha=1.0))
    model.fit(train[features], train[target])

    Path("runs").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{Path('runs/mlflow.db').resolve().as_posix()}")
    experiment_name = "Synthetic monthly incident counts"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=Path("runs/artifacts").resolve().as_uri(),
        )
    mlflow.set_experiment(experiment_name)
    checksum = hashlib.sha256(dataset_bytes).hexdigest()
    candidates = [
        ("previous-month", {}, test["previous_count"]),
        ("ridge", {"alpha": 1.0}, model.predict(test[features])),
    ]
    for name, parameters, prediction in candidates:
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({
                "model": name,
                "test_start_period": CUTOFF,
                "train_rows": len(train),
                "test_rows": len(test),
                **parameters,
            })
            mlflow.set_tags({
                "dataset.path": DATA.as_posix(),
                "dataset.sha256": checksum,
                "dataset.kind": "synthetic",
            })
            mlflow.log_artifact("dvc.lock")
            mae = mean_absolute_error(test[target], prediction)
            mlflow.log_metric("test_mae", mae)
            print(f"{name}: test MAE {mae:.3f}, run {run.info.run_id}")


if __name__ == "__main__":
    main()
