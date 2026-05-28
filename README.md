# Polygraph DSS (СППР полиграфолога)

Проект реализует базовую систему поддержки принятия решений для анализа полиграмм:
- сбор данных из `DOCX`/`XDEX`,
- подготовка признаков,
- обучение и сравнение моделей,
- инференс по новым данным,
- API и web-интерфейс.

## 1) Что требует ЭПП и что реализовано

По заданию `Задание на ЭПП_18.01.docx` требуются:
- полный пайплайн данных и ML,
- обученные модели,
- метрики качества и сравнение,
- описание архитектуры,
- отчет/презентация.

В текущей версии реализовано:
- `src/parser.py`: парсинг `DOCX` и базовый парсинг `XDEX/XDX`,
- `src/preprocess.py`: формирование `long` и `wide` датасетов,
- `src/train.py`: обучение `LogReg` и `RandomForest`, расчет метрик,
- `src/predict.py`: инференс и рекомендации,
- `api/app.py`: endpoint `/analyze`,
- `ui/streamlit_app.py`: UI для загрузки файла и просмотра результатов.

## 2) Структура

```text
Курсовая/
├── api/
│   └── app.py
├── data/
│   ├── raw/
│   │   ├── docx/
│   │   └── xdex/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── parser.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── metrics.py
│   └── run_pipeline.py
├── ui/
│   └── streamlit_app.py
└── requirements.txt
```

## 3) Запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3.1) Маршрут Для Обычного Пользователя (Один XDEX)

Если у вас на руках только один файл, например `1.xdex`, используйте:

```powershell
.\run_single_xdex.ps1 -XdexPath "C:\Users\kirill.yakupov.000\Downloads\Telegram Desktop\1.xdex"
```

Что вы получите:
- `reports/single_xdex/1/report.html` — наглядный отчет (открывается в браузере),
- `reports/single_xdex/1/question_route.png` — красивая схема маршрута вопросов,
- `reports/single_xdex/1/questions.csv` — табличка вопросов и типов,
- `reports/single_xdex/1/summary.json` — краткая сводка.

Важно:
- этот шаг показывает структуру теста и каналы,
- для ML-оценки `вероятность/рекомендация` нужен экспорт метрик из Дианы (`I/II/Результат`) в `DOCX/CSV/XLSX`.

Положить исходные файлы:
- `DOCX` в `data/raw/docx/`
- `XDEX/XDX` в `data/raw/xdex/`

Запуск полного пайплайна:

```powershell
python src/run_pipeline.py
```

Отдельно инференс:

```powershell
python src/predict.py --input-features data/processed/polygram_features.csv --model-path models/best_model.pkl
```

Построить график вероятностей по вопросам:

```powershell
python src/plot_results.py
```

API:

```powershell
uvicorn api.app:app --reload
```

UI:

```powershell
streamlit run ui/streamlit_app.py
```

UI (shortcut):

```powershell
.\run_interface.ps1
```

Desktop shortcut:

```powershell
powershell -ExecutionPolicy Bypass -File .\create_desktop_shortcut.ps1
```

After that use `Polygraph DSS.lnk` on Desktop (double-click).

Docker Compose:

```powershell
docker compose up --build
```

Сервисы:
- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## 4) Выходные артефакты

После `python src/run_pipeline.py` формируются:
- `data/processed/polygram_long.csv`
- `data/processed/polygram_features.csv`
- `models/logreg.pkl`
- `models/random_forest.pkl`
- `models/best_model.pkl`
- `reports/model_comparison.csv`
- `reports/feature_importance_best.csv`
- `reports/predictions_train.csv`
- `reports/metrics_summary.json`

## 5) Текущие результаты (на ваших `тест_*.docx`)

Данные:
- 239 строк в `long`,
- 62 объектов в `features`,
- 25 числовых признаков.

Сравнение моделей (`reports/model_comparison.csv`):
- `RandomForest`: `cv_f1=0.949`, `cv_roc_auc=1.0`
- `LogReg`: `cv_f1=0.858`, `cv_roc_auc=0.949`

Выбранная модель: `RandomForest`.

## 6) Ограничения

- Текущая разметка `label_weak` построена порогом по `score_result` (`>=0.55`), это временный surrogate.
- Объем данных пока мал для устойчивой оценки и генерализации.
- Для финальной сдачи нужна экспертная разметка (ground truth) и расширение набора тестов.

## 7) Рекомендации к следующему этапу

- Получить минимум 50+ полноценных тестов с экспертными метками.
- Проверить и откалибровать порог/вероятности.
- Добавить hold-out тест и калибровку вероятностей (`CalibratedClassifierCV`).
- Подготовить презентацию с архитектурой, метриками и интерпретацией важных каналов.

## 8) Tests And Reproducibility

Unit and regression tests are stored in `tests/`:
- parser unit tests for DOCX/XDEX (`tests/test_parsers.py`);
- model bundle format regression tests (`tests/test_model_bundle_format.py`).

Run tests:

```powershell
python -m pytest -q
```

Single entry point for full reproducible run (tests + preprocess + train + inference):

```powershell
.\run_all.ps1
```

Optional `make` targets (if `make` is available in your environment):

```powershell
make test
make pipeline
make infer
make all
```
