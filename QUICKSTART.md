# Быстрый старт - RAG Pipeline для хакатона

## Структура проекта

```
rag-alpha-hack-2025/
├── artifacts/
│   ├── data/               # Данные
│   │   ├── websites.csv    # База веб-страниц (1937 документов)
│   │   ├── questions_clean.csv  # Вопросы (6977 шт.)
│   │   └── sample_submission.csv
│   └── indices/            # Индексы (векторная БД будет здесь)
├── src/
│   ├── config.py           # Конфигурация
│   ├── data_loader.py      # Загрузка CSV данных
│   ├── preprocessing/      # Стратегии чанкинга
│   │   ├── base.py         # Базовый класс
│   │   └── chunking.py     # Реализации (NoChunker, RecursiveChunker, TitleAwareChunker)
│   └── rag/
│       ├── embeddings.py   # Модель эмбеддингов
│       └── vector_db.py    # ChromaDB векторная база
├── scripts/
│   ├── build_index.py      # Построение индекса
│   └── generate_submission.py  # Генерация submission
└── submissions/            # Выходные файлы
```

## Шаг за шагом: Запуск baseline

### 1. Активировать виртуальное окружение

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Проверить, что все библиотеки установлены

```powershell
pip list | findstr -i "sentence-transformers chromadb pandas loguru langchain"
```

Если чего-то не хватает:

```powershell
pip install sentence-transformers chromadb pandas loguru langchain tqdm
```

### 3. Построить векторный индекс

```powershell
python scripts/build_index.py
```

Это:
- Загрузит 1937 веб-страниц из `websites.csv`
- Обработает их (по умолчанию без чанкинга - каждая страница целиком)
- Создаст эмбеддинги с помощью sentence-transformers
- Сохранит в ChromaDB

**Опционально:** Можно выбрать стратегию чанкинга:

```powershell
# Без чанкинга (по умолчанию, быстрее)
python scripts/build_index.py no_chunking

# Рекурсивное разбиение (1000 токенов, overlap 200)
python scripts/build_index.py recursive

# С добавлением title к каждому чанку (800 токенов, overlap 150)
python scripts/build_index.py title_aware
```

### 4. Сгенерировать submission файл

```powershell
python scripts/generate_submission.py
```

Это:
- Загрузит 6977 вопросов
- Для каждого найдет топ-5 релевантных веб-страниц
- Сохранит в `submissions/submission_YYYYMMDD_HHMMSS.csv`

**Опционально:** Указать имя файла:

```powershell
python scripts/generate_submission.py baseline_v1.csv
```

### 5. Проверить результат

```powershell
# Посмотреть на submission
python -c "import pandas as pd; df = pd.read_csv('submissions/submission_<timestamp>.csv'); print(df.head()); print(f'\nShape: {df.shape}')"
```

## Текущая конфигурация (src/config.py)

```python
model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
vector_db_path: "../artifacts/data/vectordb"
```

### Смена модели эмбеддингов

Отредактируйте `src/config.py` или создайте `.env` файл:

```env
MODEL_NAME=intfloat/multilingual-e5-large
```

**Рекомендуемые модели для русского языка:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (быстрая, ~120MB)
- `intfloat/multilingual-e5-large` (лучше качество, ~2GB)
- `cointegrated/rubert-tiny2` (маленькая, для русского)

## Дальнейшие улучшения

1. **Hybrid Search**: BM25 + vector search
2. **Reranking**: Добавить reranker модель
3. **Лучшие эмбеддинги**: Попробовать модели специально для русского
4. **Preprocessing**: Очистка текста, удаление стоп-слов
5. **Query expansion**: Расширение запросов синонимами
6. **Оптимизация чанкинга**: Настройка chunk_size и overlap

## Troubleshooting

### Ошибка импорта langchain

```powershell
pip install langchain
```

### Ошибка с encodings (в Windows)

Если возникают проблемы с кодировкой при чтении CSV:

```python
df = pd.read_csv('file.csv', encoding='utf-8-sig')
```

### Медленная генерация эмбеддингов

Можно включить GPU (если есть CUDA):

```powershell
pip install sentence-transformers[gpu]
```

## Быстрый тест в ноутбуке

```python
from src.data_loader import DataLoader
from src.rag.vector_db import VectorDatabase

# Загрузить данные
loader = DataLoader()
stats = loader.get_data_stats()
print(stats)

# Подключиться к БД
db = VectorDatabase()
info = db.get_collection_info()
print(info)

# Тестовый поиск
results = db.search("Как получить ипотеку?", n_results=5)
for i, r in enumerate(results, 1):
    print(f"{i}. web_id: {r['web_id']}, title: {r['metadata']['title'][:50]}...")
```

