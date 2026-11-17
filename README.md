# MLOps — практические работы №3 и №4

## Требования к окружению

- Python 3.12
- Poetry

## Установка окружения

Клонируйте репозиторий и перейдите в его папку:

```bash
git clone https://github.com/aslllnv/mlops-pract3-vostrikova.git
cd mlops-pract3-vostrikova
```

Установите зависимости:

```bash
poetry install
```

Активируйте окружение через команды с `poetry run`.

## Восстановление данных

Для восстановления данных Wine Dataset используется ClearML Dataset.

ID датасета:

```text
0b666c28b1fe486d811fb1346e7b43eb
```

Версия: `1.1.1`.

SHA-256:

```text
724d80b05e39c7a72fe7631de59583c764049137f9d7089b01b5460e212c75c0
```

## Подготовка данных

```bash
poetry run python prepare_data.py
```

## Запуск MLflow

Локальный MLflow использует SQLite:

```bash
poetry run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Адрес интерфейса:

```text
http://127.0.0.1:5000
```

## Запуск экспериментов — практическая работа №4

В отдельном терминале из корня проекта выполните:

```bash
poetry run python run_experiments.py \
  --tracking-uri "sqlite:///mlflow.db" \
  --dataset-id "0b666c28b1fe486d811fb1346e7b43eb" \
  --dataset-sha256 "724d80b05e39c7a72fe7631de59583c764049137f9d7089b01b5460e212c75c0"
```

Скрипт загружает указанный датасет, проверяет его SHA-256, запускает конфигурации моделей, выбирает лучшую по validation F1 и оценивает её на тестовой выборке.

Повторный запуск создаёт новые runs в том же MLflow experiment.

## Результаты

Подробное описание экспериментов, метрики, идентификаторы MLflow и ClearML, а также результаты проверки модели приведены в [REPORT.md](REPORT.md).

Файл `verification.json` содержит результаты проверки модели.