## Структура проекта

```
├── app.py                      # CLI для предсказания зарплат
├── pipeline_app.py             # Пайплайн обработки CSV -> npy
├── model/
│   ├── __init__.py
│   ├── regressor.py            # Ridge регрессия
│   └── nn_regressor.py         # FCN нейронная сеть (PyTorch)
├── scripts/
│   ├── train.py                # Обучение Ridge модели
│   └── train_nn.py             # Обучение FCN с MLflow трекингом
├── pipeline/
│   ├── base_handler.py         # Базовый класс обработчика
│   └── handlers.py             # Обработчики данных
├── resources/
│   ├── model_weights.joblib    # Веса Ridge модели
│   └── nn_model_weights.pt     # Веса FCN модели
├── example/
│   ├── hh.csv                  # Пример исходных данных
│   ├── x_data.npy              # Обработанные признаки
│   └── y_data.npy              # Целевые значения (зарплаты)
└── requirements.txt
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Предсказание зарплат (CLI)

```bash
python3 app.py path/to/x_data.npy
```

Принимает на вход файл с признаками (выход пайплайна) и выводит список предсказанных зарплат в рублях (float), по одному значению на строку.

**Пример:**
```bash
python3 app.py example/x_data.npy
```

**Вывод:**
```
17094.65
96361.02
82976.52
98835.98
...
```

### Обработка данных (пайплайн)

Преобразование исходного CSV с hh.ru в numpy-массивы:

```bash
python3 pipeline_app.py path/to/hh.csv
```

Создаёт два файла в директории с исходным CSV:
- `x_data.npy` — матрица признаков `(n_samples, 76)`
- `y_data.npy` — вектор зарплат `(n_samples,)` в тысячах рублей

### Обучение Ridge модели

```bash
python3 scripts/train.py path/to/x_data.npy path/to/y_data.npy
```

Обучает Ridge регрессию, выводит метрики, сохраняет веса в `resources/model_weights.joblib`.

### Обучение FCN нейронной сети с MLflow трекингом

```bash
python3 scripts/train_nn.py path/to/x_data.npy path/to/y_data.npy
```

Обучает полносвязную нейронную сеть (FCN) и логирует эксперимент в MLflow:
- **Tracking URI:** `http://kamnsv.com:55000/`
- **Experiment:** `LIne Regression HH`
- **Model name:** `kholev_artem_fcn`
- **Метрика:** `r2_score_test`

Сохраняет веса в `resources/nn_model_weights.pt`.

## Модели

### Ridge Regression

Линейная регрессия с L2-регуляризацией (`alpha=1.0`).

### FCN (Fully Connected Network)

Полносвязная нейронная сеть на PyTorch:

```
Input(76) → Linear(256) → BatchNorm → ReLU → Dropout(0.1)
          → Linear(256) → BatchNorm → ReLU → Dropout(0.1)
          → Linear(128) → BatchNorm → ReLU → Dropout(0.1)
          → Linear(1)
```

**Параметры обучения:**
- Оптимизатор: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=15
- Batch size: 64

**Метрики на тестовой выборке (FCN):**

| Метрика | Значение |
|---------|----------|
| R²      | ~0.26    |
| MAE     | ~42 000 руб |
| RMSE    | ~77 000 руб |

**Формат данных:** зарплаты в `y_data.npy` хранятся в тысячах рублей. При предсказании и логировании метрик значения конвертируются в рубли (×1000).

## Признаки

Модель использует 76 признаков после one-hot encoding:

| Признак | Описание |
|---------|----------|
| Age | Возраст кандидата |
| Experience_Years | Опыт работы в годах |
| Gender | Пол (male/female) |
| City | Город (с группировкой редких в "other") |
| Employment | Тип занятости (full/part) |
| Schedule | График работы (full_day/flexible/remote/shift) |
| Education | Образование (higher/vocational/secondary) |
| Has_Car | Наличие автомобиля |

## Пайплайн обработки данных

Данные проходят через цепочку обработчиков (Chain of Responsibility):

1. **DataLoaderHandler** — загрузка CSV
2. **DataCleaningHandler** — удаление дубликатов, очистка текста
3. **FeatureExtractionHandler** — извлечение признаков из сырых данных
4. **MissingDataHandler** — заполнение пропущенных значений
5. **OutlierRemovalHandler** — удаление выбросов (IQR метод)
6. **CategoryGroupingHandler** — группировка редких категорий
7. **EncodingHandler** — one-hot encoding категорий
8. **NormalizationHandler** — нормализация (StandardScaler)
9. **ArrayConversionHandler** — конвертация в numpy-массивы
