from pathlib import Path
import argparse

from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")


def load_dataset(name):
    if name == "iris":
        dataset = load_iris(as_frame=True)
    else:
        dataset = load_wine(as_frame=True)

    df = dataset.frame.copy()

    # Сортируем по всем исходным признакам и target.
    df = df.sort_values(
        by=list(df.columns),
        kind="mergesort",
        ignore_index=True,
    )

    return df


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["base", "wine", "iris_features"],
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "base":
        iris = load_dataset("iris")
        save_csv(iris, DATA_DIR / "iris.csv")

    elif args.mode == "wine":
        wine = load_dataset("wine")
        save_csv(wine, DATA_DIR / "wine.csv")

    elif args.mode == "iris_features":
        iris = load_dataset("iris")

        iris["sepal_area"] = (
            iris["sepal length (cm)"]
            * iris["sepal width (cm)"]
        )
        iris["petal_area"] = (
            iris["petal length (cm)"]
            * iris["petal width (cm)"]
        )

        save_csv(iris, DATA_DIR / "iris.csv")

    print(f"Готово: режим {args.mode}")


if __name__ == "__main__":
    main()
