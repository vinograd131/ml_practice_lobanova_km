"""Загрузка датасета Predict Students' Dropout and Academic
Success (UCI, CC BY 4.0)."""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "students_dropout.zip"
CSV_NAME = "data.csv"

EXPECTED_ROWS = 4424
EXPECTED_COLS = 37  # 36 признаков + Target
TARGET = "Target"
EXPECTED_CLASSES = {"Dropout", "Enrolled", "Graduate"}


def download(url: str, dest: Path) -> None:
    # parents=True — создать и промежуточные папки (data/, data/raw/),
    # exist_ok=True — не падать, если папка уже существует
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    print(f"Скачано: {url} -> {dest}")


def extract(zip_path: Path, member: str, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if member not in names:
            raise FileNotFoundError(
                f"В архиве {zip_path} нет файла {member!r}. Есть: {names}"
            )
        zf.extract(member, out_dir)
    return out_dir / member


def validate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise ValueError(f"Файл не найден: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"Файл пустой: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    n_rows, n_cols = df.shape
    if n_rows != EXPECTED_ROWS:
        raise ValueError(f"Ожидалось {EXPECTED_ROWS} строк, получено {n_rows}")
    if n_cols != EXPECTED_COLS:
        raise ValueError(f"Ожидалось {EXPECTED_COLS} колонок, получено {n_cols}")

    if TARGET not in df.columns:
        raise ValueError(f"Нет целевой колонки {TARGET!r}")
    classes = set(df[TARGET].unique())
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Классы в {TARGET!r}: {classes}, ожидались {EXPECTED_CLASSES}"
        )

    return df


def main() -> None:
    download(URL, ZIP_PATH)
    csv_path = extract(ZIP_PATH, CSV_NAME, RAW_DIR)
    df = validate(csv_path)
    print(f"OK: {csv_path} — {df.shape[0]} строк, {df.shape[1]} колонок")


if __name__ == "__main__":
    main()
