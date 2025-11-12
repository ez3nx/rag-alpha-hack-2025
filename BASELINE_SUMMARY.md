# 🎯 Baseline RAG Pipeline - Результаты

## ✅ Что реализовано

### 1. **Архитектура системы**

```
src/
├── config.py                 # Центральная конфигурация
├── data_loader.py            # Загрузка CSV данных
├── preprocessing/
│   ├── base.py              # Базовый класс для чанкинга
│   └── chunking.py          # 3 стратегии: NoChunker, RecursiveChunker, TitleAwareChunker
└── rag/
    ├── embeddings.py        # Wrapper для sentence-transformers
    └── vector_db.py         # ChromaDB векторная база

scripts/
├── build_index.py           # Построение индекса
└── generate_submission.py   # Генерация результатов
```

### 2. **Технический стек**

- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector DB**: ChromaDB (cosine similarity)
- **Chunking**: Langchain RecursiveCharacterTextSplitter
- **Data**: 1937 веб-страниц, 6977 вопросов

### 3. **Baseline результаты**

✅ **Submission файл**: `submissions/baseline_v1.csv`
- ✅ Все 6977 вопросов обработаны
- ✅ Каждый вопрос имеет ровно 5 результатов
- ⏱️ Время построения индекса: ~1.5 мин
- ⏱️ Время генерации submission: ~4.5 мин

**Топ-10 наиболее релевантных документов:**
1. web_id 1157: 484 раза
2. web_id 16: 449 раз
3. web_id 1159: 403 раза
4. web_id 878: 355 раз
5. web_id 1567: 349 раз

---

## 🚀 Как запустить

### Быстрый старт

```powershell
# 1. Активировать окружение
.\venv\Scripts\Activate.ps1

# 2. Построить индекс (по умолчанию без чанкинга)
python scripts/build_index.py

# 3. Сгенерировать submission
python scripts/generate_submission.py baseline_v1.csv
```

### Стратегии чанкинга

```powershell
# Без чанкинга (быстро, каждая страница целиком)
python scripts/build_index.py no_chunking

# Рекурсивное разбиение (chunk_size=1000, overlap=200)
python scripts/build_index.py recursive

# С добавлением title к каждому чанку (chunk_size=800, overlap=150)
python scripts/build_index.py title_aware
```

---

## 📊 Следующие шаги для улучшения Hit@5

### Приоритет 1: Лучшие эмбеддинги (Expected: +5-10% Hit@5)

**Рекомендуемые модели:**

```python
# В src/config.py или .env файле
MODEL_NAME = "intfloat/multilingual-e5-large"  # Лучшее качество для multilingual
# или
MODEL_NAME = "sentence-transformers/LaBSE"     # Отличная для многоязычности
# или  
MODEL_NAME = "cointegrated/rubert-tiny2"       # Специально для русского
```

**Как протестировать:**
1. Изменить `MODEL_NAME` в `src/config.py`
2. Удалить старый индекс: `rm -rf artifacts/vectordb/`
3. Пересобрать индекс: `python scripts/build_index.py`
4. Сгенерировать новый submission

### Приоритет 2: Hybrid Search - BM25 + Dense (Expected: +10-15% Hit@5)

**План:**
- Добавить BM25 поиск (keyword-based)
- Комбинировать с vector search (semantic)
- Использовать Reciprocal Rank Fusion (RRF)

**Псевдокод:**
```python
# 1. BM25 поиск
bm25_results = bm25_index.search(query, top_k=10)

# 2. Vector поиск  
vector_results = vector_db.search(query, top_k=10)

# 3. Fusion
final_results = reciprocal_rank_fusion([bm25_results, vector_results], k=5)
```

### Приоритет 3: Reranking (Expected: +5-8% Hit@5)

**План:**
- Добавить Cross-Encoder для reranking топ-20 результатов
- Модели: `cross-encoder/ms-marco-MiniLM-L-12-v2` или `BAAI/bge-reranker-base`

**Архитектура:**
```
Query → Vector Search (top 20) → Reranker → Final top 5
```

### Приоритет 4: Query Enhancement

**Техники:**
1. **Query Expansion**: добавление синонимов
2. **Query Rewriting**: переформулирование с помощью LLM
3. **Multi-Query**: генерация нескольких вариантов запроса

### Приоритет 5: Улучшение препроцессинга

**Текущие проблемы:**
- Возможны артефакты парсинга HTML
- Нет нормализации текста
- Title и text обрабатываются одинаково

**Улучшения:**
```python
# Очистка текста
text = remove_html_artifacts(text)
text = normalize_whitespace(text)
text = fix_encoding_issues(text)

# Взвешивание title vs text
embedding = 0.7 * title_embedding + 0.3 * text_embedding
```

### Приоритет 6: Оптимизация чанкинга

**Эксперименты:**
- [ ] Разные размеры chunk_size: [500, 800, 1000, 1500, 2000]
- [ ] Разные overlap: [0, 100, 150, 200, 300]
- [ ] Semantic chunking (по смыслу, а не по размеру)
- [ ] Агрегация чанков при поиске (разные стратегии)

---

## 🔧 Полезные команды

### Проверка индекса

```python
from src.rag.vector_db import VectorDatabase

db = VectorDatabase()
info = db.get_collection_info()
print(f"Документов в индексе: {info['count']}")

# Тестовый поиск
results = db.search("Как получить ипотеку?", n_results=5)
for i, r in enumerate(results, 1):
    print(f"{i}. web_id: {r['web_id']}, title: {r['metadata']['title'][:50]}...")
```

### Статистика данных

```python
from src.data_loader import DataLoader

loader = DataLoader()
stats = loader.get_data_stats()
print(stats)
```

### Очистка индекса

```powershell
# Удалить векторную базу (для пересборки)
rm -rf artifacts/vectordb/
```

---

## 📈 Метрики для отслеживания

1. **Hit@5**: Основная метрика (хотя бы 1 релевантный в топ-5)
2. **MRR@5**: Mean Reciprocal Rank (позиция первого релевантного)
3. **NDCG@5**: Normalized Discounted Cumulative Gain
4. **Latency**: Время обработки одного запроса
5. **Coverage**: Сколько уникальных документов попало в результаты

---

## 🎓 Рекомендации по экспериментам

### Быстрые эксперименты (1-2 часа)
1. ✅ Разные модели эмбеддингов
2. ✅ Разные стратегии чанкинга
3. ✅ Настройка n_results (брать больше результатов и агрегировать)

### Средние эксперименты (4-6 часов)
1. 🔄 BM25 + Hybrid Search
2. 🔄 Reranking модель
3. 🔄 Query expansion

### Сложные эксперименты (1-2 дня)
1. 🔄 Ensemble из нескольких моделей
2. 🔄 Fine-tuning эмбеддинг модели на ваших данных
3. 🔄 Metadata filtering (использование поля `kind`)

---

## 📝 Чеклист перед submission

- [ ] Проверить формат CSV (q_id, web_list)
- [ ] Убедиться, что все вопросы имеют ровно 5 результатов
- [ ] Проверить, что web_id существуют в websites.csv
- [ ] Запустить на полном датасете (все 6977 вопросов)
- [ ] Сохранить версию индекса и параметры

---

## 🏆 Текущий статус

✅ **Baseline готов и запущен!**
- Индекс построен: 1937 документов
- Submission сгенерирован: 6977 вопросов × 5 результатов
- Файл: `submissions/baseline_v1.csv`

**Следующий шаг:** Начать эксперименты с улучшениями (см. приоритеты выше)

---

## 💡 Дополнительные идеи

1. **Анализ ошибок**: Посмотреть на вопросы, где baseline fail
2. **Visualization**: t-SNE/UMAP эмбеддингов для понимания clusters
3. **AB testing**: Сравнивать разные подходы на validation set
4. **Ensemble**: Комбинировать несколько моделей через voting
5. **Active Learning**: Использовать feedback для улучшения

---

Готов побеждать? Давай экспериментировать! 🚀

