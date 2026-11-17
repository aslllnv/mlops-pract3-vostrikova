import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from clearml import Dataset
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_NAME = "miit-mlops-practice-4-vostrikova"
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Воспроизводимое сравнение моделей Wine в MLflow"
    )

    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow Tracking URI",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("WINE_ID"),
        required=os.getenv("WINE_ID") is None,
        help="ClearML Dataset ID",
    )
    parser.add_argument(
        "--dataset-sha256",
        default=os.getenv("WINE_SHA256"),
        required=os.getenv("WINE_SHA256") is None,
        help="Ожидаемый SHA-256 файла wine.csv",
    )

    return parser.parse_args()


def calculate_metrics(y_true, y_pred, prefix="val_"):
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def build_model(config):
    params = config["params"]

    if config["family"] == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**params)),
        ])

    return RandomForestClassifier(**params)


def log_model(model, input_data):
    input_example = input_data.head(5)
    predictions = model.predict(input_example)
    signature = infer_signature(input_example, predictions)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=input_example,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
    )



def check_run_logging(client, run_id, required_params, required_metrics):
    run = client.get_run(run_id)
    data = run.data

    missing_params = [
        key for key in required_params if key not in data.params
    ]
    missing_metrics = [
        key for key in required_metrics if key not in data.metrics
    ]

    if missing_params or missing_metrics:
        raise RuntimeError(
            f"Неполное логирование run {run_id}. "
            f"Отсутствуют параметры: {missing_params}; "
            f"метрики: {missing_metrics}"
        )



def main():
    args = parse_args()

    if not args.dataset_id or not args.dataset_sha256:
        raise ValueError(
            "Необходимо передать Dataset ID и SHA-256 "
            "через аргументы или переменные окружения."
        )

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Загрузка датасета и проверка SHA-256
    dataset = Dataset.get(dataset_id=args.dataset_id)
    dataset_path = Path(
        dataset.get_local_copy(use_soft_links=False)
    )
    wine_file = dataset_path / "wine.csv"

    if not wine_file.is_file():
        raise FileNotFoundError(f"Файл не найден: {wine_file}")

    actual_sha256 = hashlib.sha256(
        wine_file.read_bytes()
    ).hexdigest()

    if actual_sha256.lower() != args.dataset_sha256.lower():
        raise ValueError(
            "SHA-256 не совпадает! "
            f"Ожидалось: {args.dataset_sha256}, "
            f"получено: {actual_sha256}"
        )

    df = pd.read_csv(wine_file)

    if "target" not in df.columns:
        raise ValueError("В wine.csv отсутствует столбец target")

    X = df.drop(columns=["target"])
    y = df["target"]

    print("Данные Wine загружены и проверены.")
    print("Размер таблицы:", df.shape)
    print("SHA-256:", actual_sha256)

    # Разделение 60/20/20 со стратификацией
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("Размеры выборок:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # Создание experiment, если он отсутствует
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        print("Создан новый experiment.")
    else:
        experiment_id = experiment.experiment_id
        print("Используется существующий experiment.")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Все шесть конфигураций
    model_configs = [
        {
            "name": "baseline-logistic-regression",
            "family": "LogisticRegression",
            "params": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "logistic-regression-c0.1",
            "family": "LogisticRegression",
            "params": {
                "C": 0.1,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "logistic-regression-c10",
            "family": "LogisticRegression",
            "params": {
                "C": 10.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "random-forest-100",
            "family": "RandomForest",
            "params": {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "random-forest-200-depth5",
            "family": "RandomForest",
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "random-forest-300-depth10",
            "family": "RandomForest",
            "params": {
                "n_estimators": 300,
                "max_depth": 10,
                "random_state": RANDOM_STATE,
            },
        },
    ]

    # Уникальный идентификатор текущего выполнения
    execution_id = str(uuid.uuid4())
    current_run_ids = []
    original_models = {}

    # Обучение и логирование всех конфигураций
    for config in model_configs:
        model = build_model(config)

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        val_predictions = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, val_predictions)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=config["name"],
        ) as run:
            run_id = run.info.run_id
            current_run_ids.append(run_id)
            original_models[run_id] = model

            mlflow.set_tags({
                "model_family": config["family"],
                "dataset_id": args.dataset_id,
                "dataset_sha256": actual_sha256,
                "execution_id": execution_id,
            })

            mlflow.log_params(config["params"])
            mlflow.log_param("model_type", config["family"])

            mlflow.log_metrics(val_metrics)
            mlflow.log_metric("training_time", training_time)

            log_model(model, X_train)

        print(
            f"Run завершён: {config['name']}; "
            f"Run ID: {run_id}; "
            f"Validation F1: {val_metrics['val_f1_weighted']:.6f}"
        )

    if len(current_run_ids) != 6:
        raise RuntimeError("В текущем выполнении создано не шесть runs.")

    # Получение таблицы только текущего выполнения
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.execution_id = '{execution_id}'",
        order_by=["metrics.val_f1_weighted DESC"],
    )

    runs = runs.dropna(subset=["metrics.val_f1_weighted"])

    if len(runs) != 6:
        raise RuntimeError(
            f"Ожидалось 6 текущих runs, найдено: {len(runs)}"
        )

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    print("\nЛучшая модель выбрана по validation F1.")
    print("Best Run ID:", best_run_id)
    print("Validation F1:", best_run["metrics.val_f1_weighted"])

    # Проверка логирования до оценки на test
    required_val_metrics = [
        "val_accuracy",
        "val_precision_weighted",
        "val_recall_weighted",
        "val_f1_weighted",
        "training_time",
    ]

    for run_id in current_run_ids:
        run = client.get_run(run_id)
        family = run.data.tags.get("model_family")

        required_params = ["model_type", "random_state"]

        if family == "LogisticRegression":
            required_params += ["C", "solver", "max_iter"]
        elif family == "RandomForest":
            required_params += ["n_estimators", "max_depth"]
        else:
            raise RuntimeError(
                f"Неизвестное семейство модели: {family}"
            )

        check_run_logging(
            client,
            run_id,
            required_params,
            required_val_metrics,
        )

    # Загрузка лучшей модели из MLflow
    best_model = mlflow.sklearn.load_model(model_uri)

    # Проверка исходных имён признаков и совпадения предсказаний
    if list(X_test.columns) != list(X.columns):
        raise RuntimeError("Имена признаков test не совпадают с исходными.")

    original_model = original_models[best_run_id]

    original_predictions = original_model.predict(X_test)
    loaded_predictions = best_model.predict(X_test)

    predictions_match = bool(
        (original_predictions == loaded_predictions).all()
    )

    if not predictions_match:
        raise RuntimeError(
            "Предсказания исходной и загруженной моделей не совпадают."
        )

    model_info = mlflow.models.get_model_info(model_uri)
    signature_present = model_info.signature is not None

    if not signature_present:
        raise RuntimeError("У модели отсутствует signature.")

    # Test оценивается только после выбора лучшей модели
    test_metrics = calculate_metrics(
        y_test,
        best_model.predict(X_test),
        prefix="test_",
    )

    client.log_batch(
        best_run_id,
        metrics=[
            mlflow.entities.Metric(
                key=key,
                value=float(value),
                timestamp=int(time.time() * 1000),
                step=0,
            )
            for key, value in test_metrics.items()
        ],
    )

    # Проверка test-метрик после логирования
    check_run_logging(
        client,
        best_run_id,
        [],
        required_val_metrics + list(test_metrics.keys()),
    )

    # Сохранение verification.json
    verification = {
        "dataset_id": args.dataset_id,
        "dataset_sha256": actual_sha256,
        "experiment_id": experiment_id,
        "execution_id": execution_id,
        "current_run_ids": current_run_ids,
        "best_run_id": best_run_id,
        "model_uri": model_uri,
        "model_family": best_run["tags.model_family"],
        "test_uses_original_feature_names": True,
        "original_and_loaded_predictions_match": predictions_match,
        "signature_present": signature_present,
        "required_params_present": True,
        "required_metrics_present": True,
    }

    verification_path = Path("verification.json")

    with verification_path.open("w", encoding="utf-8") as file:
        json.dump(verification, file, ensure_ascii=False, indent=4)

    client.log_artifact(best_run_id, str(verification_path))

    print("\nЭксперименты успешно завершены.")
    print("Experiment ID:", experiment_id)
    print("Best Run ID:", best_run_id)
    print("Test F1:", test_metrics["test_f1_weighted"])
    print("Verification:", verification_path.resolve())


if __name__ == "__main__":
    main()
