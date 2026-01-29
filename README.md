## Структура проекта

```
├── app.py                  # CLI для предсказания зарплат
├── pipeline_app.py         # Пайплайн обработки CSV -> npy
├── model/
│   ├── __init__.py
│   └── regressor.py        # Класс Ridge регрессии
├── scripts/
│   └── train.py            # Скрипт обучения модели
├── pipeline/
│   ├── base_handler.py     # Базовый класс обработчика
│   └── handlers.py         # Обработчики данных
├── resources/
│   └── model_weights.joblib    # Сохранённые веса модели
├── example/
│   ├── hh.csv              # Пример исходных данных
│   ├── x_data.npy          # Обработанные признаки
│   └── y_data.npy          # Целевые значения (зарплаты)
└── requirements.txt
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Основной интерфейс - предсказание зарплат

```bash
python3 app.py path/to/x_data.npy
```

Принимает на вход файл с признаками (выход пайплайна) и выводит список предсказанных зарплат в рублях (float), по одному значению на строку

**Пример:**
```bash
python app.py example/x_data.npy
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

Преобразование исходного CSV файла с hh.ru в numpy массивы:

```bash
python3 pipeline_app.py path/to/hh.csv
```

Создаёт два файла в директории с исходным CSV:
- `x_data.npy` - матрица признаков (n_samples, 76)
- `y_data.npy` - вектор зарплат (n_samples,)

### Обучение модели

Обучение модели на подготовленных данных:

```bash
python3 scripts/train.py path/to/x_data.npy path/to/y_data.npy
```

Модель автоматически:
- Разбивает данные на train/test (70%/30%)
- Обучает Ridge регрессию
- Выводит метрики качества
- Сохраняет веса в `resources/model_weights.joblib`

## Модель

**Алгоритм:** Ridge Regression (линейная регрессия с L2-регуляризацией)

**Формат данных:** в `y_data.npy` зарплаты хранятся в тысячах рублей (60.0 = 60 000 руб). При выводе предсказаний и метрик значения конвертируются в рубли (умножаются на 1000)

**Метрики на тестовой выборке:**
| Метрика | Значение | Описание |
|---------|----------|----------|
| R²      | 0.25     | Коэффициент детерминации - модель объясняет 25% дисперсии данных |
| MAE     | ~44 000 руб | Mean Absolute Error - средняя ошибка предсказания |
| RMSE    | ~77 000 руб | Root Mean Square Error - среднеквадратичная ошибка (сильнее штрафует большие отклонения) |

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

1. **DataLoaderHandler** - загрузка CSV
2. **DataCleaningHandler** - удаление дубликатов, очистка текста
3. **FeatureExtractionHandler** - извлечение признаков из сырых данных
4. **MissingDataHandler** - заполнение пропущенных значений
5. **OutlierRemovalHandler** - удаление выбросов (IQR метод)
6. **CategoryGroupingHandler** - группировка редких категорий
7. **EncodingHandler** - one-hot encoding категорий
8. **NormalizationHandler** - нормализация (StandardScaler)
9. **ArrayConversionHandler** - конвертация в numpy массивы
