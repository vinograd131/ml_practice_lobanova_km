# Прогноз отчисления студентов

Учебный воспроизводимый ML-проект. По демографическим, социально-экономическим и
академическим признакам студента модель предсказывает исход обучения: отчислится
(`Dropout`), продолжит учиться (`Enrolled`) или выпустится (`Graduate`).

Модель: `StandardScaler` + `LogisticRegression` (scikit-learn), для сравнения
используется baseline `DummyClassifier`, который всегда предсказывает самый частый класс.

## Данные

- **Датасет:** Predict Students' Dropout and Academic Success
- **Источник:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success),
  DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)
- **Лицензия:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Описание:** 4 424 студента, 36 признаков, целевая колонка `Target` с тремя
  классами. Пропусков нет. Классы несбалансированы: Graduate ≈ 50 %,
  Dropout ≈ 32 %, Enrolled ≈ 18 %.

**Токены и регистрация для загрузки не нужны.**

## Установка

Нужен [uv](https://docs.astral.sh/uv/getting-started/installation/). Подходящую
версию Python (3.13+) uv скачает сам, если её нет в системе.

```powershell
git clone https://github.com/vinograd131/ml_practice_lobanova_km.git
cd ml_practice_lobanova_km
uv sync
```

`uv sync` создаёт виртуальное окружение `.venv` и ставит точные версии всех
зависимостей из `uv.lock`, включая dev-инструменты ruff, pre-commit.

## Загрузка данных

```powershell
uv run python -m src.get_data
```

Скрипт:

1. скачивает zip-архив датасета с UCI в `data/raw/`;
2. распаковывает из него `data.csv`;
3. проверяет файл: он существует и не пустой, в нём 4 424 строки и 37 колонок,
   есть колонка `Target` ровно с тремя классами `Dropout`, `Enrolled`, `Graduate`.

Если какая-то проверка не прошла, скрипт завершается с ошибкой. Папка `data/`
не хранится в Git: данные всегда скачиваются заново.

## Проверка кода

Настройки Ruff лежат в `pyproject.toml` в `[tool.ruff]`, хуки
pre-commit находятся в `.pre-commit-config.yaml`. Там базовые проверки файлов, `ruff check`
и `ruff format`.

Один раз после клонирования установите git-хук, чтобы проверки запускались
автоматически перед каждым `git commit`:

```powershell
uv run pre-commit install
```

Запустить все проверки вручную по всему проекту:

```powershell
uv run pre-commit run --all-files
```

Только линтер или только проверка форматирования:

```powershell
uv run ruff check .
uv run ruff format --check .
```

## Обучение

```powershell
uv run python -m src.train
```

Скрипт читает `data/raw/data.csv`, делит данные на train/test (80/20, со
стратификацией по классам, `random_state=42`), обучает модель и baseline и
сохраняет:

- `artifacts/model.pkl` это обученный пайплайн, сериализованный через `pickle`;
- `artifacts/metrics.json` это метрики `accuracy`, `f1_macro` на тестовой выборке
  для модели и baseline, время запуска, размеры выборок и параметры.

Артефакты пересоздаются при каждом запуске и не хранятся в Git. Если данных нет,
скрипт попросит сначала запустить `src.get_data`.

Ожидаемый результат (из-за фиксированного `random_state` он одинаков между запусками):

| Модель | accuracy | f1_macro |
|---|---|---|
| LogisticRegression | 0.768 | 0.683 |
| Baseline (most frequent) | 0.499 | 0.222 |

## Структура проекта

```
.
├── src/
│   ├── __init__.py
│   ├── get_data.py          # загрузка и проверка датасета
│   └── train.py             # обучение и сохранение артефактов
├── data/raw/                # скачанные данные, после отработки get_data.py
├── artifacts/               # model.pkl, metrics.json, после отработки train.py
├── reports/hw-1.md          # отчёт по ДЗ
├── pyproject.toml           # зависимости и настройки Ruff
├── uv.lock                  # зафиксированные версии зависимостей
└── .pre-commit-config.yaml  # хуки pre-commit
```
