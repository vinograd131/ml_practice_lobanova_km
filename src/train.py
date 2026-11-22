"""Обучение классификатора отчисления студентов."""

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TARGET = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла {path}. Сначала запустите: uv run python -m src.get_data"
        )
    df = pd.read_csv(path, sep=";")
    # В исходных данных у одной колонки в конце имени стоит табуляция
    df.columns = df.columns.str.strip()
    return df


def evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    df = load_data(DATA_PATH)
    x = df.drop(columns=TARGET)
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(x_train, y_train)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)

    metrics = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": "StandardScaler + LogisticRegression",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "model_metrics": evaluate(y_test, model.predict(x_test)),
        "baseline_metrics": evaluate(y_test, baseline.predict(x_test)),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Метрики сохранены: {METRICS_PATH}")


if __name__ == "__main__":
    main()
