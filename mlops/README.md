# MLflow and DVC example

This companion to the crime-pattern analysis uses a deterministic synthetic panel of monthly incident counts. It has no real crime records, identifiers or locations. Its evaluation measures the example code and experiment workflow, with no inference about real-world crime.

From a checkout, open a terminal in `mlops/` with Python 3.12 and an active virtual environment:

```sh
python -m pip install -r requirements.txt
dvc repro
python train.py
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///runs/mlflow.db
```

Open http://127.0.0.1:5000 and select **Synthetic monthly incident counts**. Compare the previous-month baseline and ridge regression on the final six months. This is one-month-ahead evaluation: each prediction can use the previous month's observed count, including during the held-out period.

DVC records the generation script and dataset checksum in `dvc.lock`. Each MLflow run logs its parameters, held-out mean absolute error, dataset SHA-256 and a copy of the DVC lock. The database and run artifacts stay local. A fresh clone regenerates the dataset with `dvc repro` from the committed generator and lock; no shared data remote is needed. Locally, `dvc checkout` can restore the generated CSV from cache.
