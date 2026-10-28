
from pathlib import Path
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def expected_data(name):
    loader = load_iris if name == "iris" else load_wine
    df = loader(as_frame=True).frame.copy()

    return df.sort_values(
        by=list(df.columns),
        kind="mergesort",
        ignore_index=True,
    )


def check_file(path, dataset_name, feature_mode=False):
    errors = []

    if not path.is_file():
        return [f"Отсутствует файл: {path}"]

    df = pd.read_csv(path)
    expected = expected_data(dataset_name)

    expected_columns = list(expected.columns)

    if feature_mode:
        expected_columns += ["sepal_area", "petal_area"]

    if any(str(col).startswith("Unnamed:") for col in df.columns):
        errors.append("Обнаружен безымянный столбец индекса")

    if len(df) != len(expected):
        errors.append(
            f"Неверное число строк: ожидалось {len(expected)}, "
            f"получено {len(df)}"
        )

    if list(df.columns) != expected_columns:
        errors.append(
            f"Неверные столбцы: {list(df.columns)}"
        )

    if list(df.columns[:len(expected.columns)]) == list(expected.columns):
        original = df.iloc[:, :len(expected.columns)]

        if not original.equals(expected):
            errors.append("Исходные признаки или target изменены")

    if feature_mode and all(
        col in df.columns
        for col in [
            "sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)",
            "sepal_area", "petal_area",
        ]
    ):
        if not np.allclose(
            df["sepal_area"],
            df["sepal length (cm)"] * df["sepal width (cm)"],
        ):
            errors.append("Неверные значения sepal_area")

        if not np.allclose(
            df["petal_area"],
            df["petal length (cm)"] * df["petal width (cm)"],
        ):
            errors.append("Неверные значения petal_area")

    print(f"\nФайл: {path.as_posix()}")
    print(f"SHA-256: {sha256(path)}")
    print(f"Строк: {df.shape[0]}")
    print(f"Столбцов: {df.shape[1]}")
    print(f"Столбцы: {df.columns.tolist()}")
    print(f"Пропущенных значений: {df.isna().sum().sum()}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["base", "wine", "iris_features", "merged"],
    )
    mode = parser.parse_args().mode

    errors = []

    iris_features = mode in ["iris_features", "merged"]

    errors.extend(
        check_file(
            DATA_DIR / "iris.csv",
            "iris",
            feature_mode=iris_features,
        )
    )

    expected_files = {
        "base": {"iris.csv"},
        "wine": {"iris.csv", "wine.csv"},
        "iris_features": {"iris.csv"},
        "merged": {"iris.csv", "wine.csv"},
    }[mode]

    actual_files = {p.name for p in DATA_DIR.glob("*.csv")}

    for name in expected_files:
        if not (DATA_DIR / name).is_file():
            errors.append(f"Отсутствует ожидаемый файл: {name}")

    for name in actual_files - expected_files:
        errors.append(f"Неожиданный файл: {name}")

    if mode in ["wine", "merged"]:
        errors.extend(
            check_file(DATA_DIR / "wine.csv", "wine")
        )

    if errors:
        print("\nОШИБКИ:")
        for error in errors:
            print("-", error)
        sys.exit(1)

    print("\nПроверка пройдена успешно!")


if __name__ == "__main__":
    main()
