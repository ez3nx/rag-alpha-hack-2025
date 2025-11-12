# 🚀 Запуск на Kaggle GPU

## Что нужно выгрузить на Kaggle

### 1. **Индексы** (обязательно!)

Выгрузить как Kaggle Dataset:

```
artifacts/
├── vectordb/           # ChromaDB векторная база (~500MB)
└── bm25_index.pkl      # BM25 индекс (~50MB)
```

**Как создать Dataset:**
1. Kaggle → Datasets → New Dataset
2. Название: `rag-indices` 
3. Загрузить папки выше
4. Сделать **Public** или **Private**

### 2. **Данные** (если ещё нет)

```
artifacts/data/
├── websites.csv
└── questions_clean.csv
```

### 3. **Код** (автоматически из Notebook)

```
src/
├── config.py
├── data_loader.py
├── preprocessing/
│   ├── __init__.py
│   ├── base.py
│   └── chunking.py
└── rag/
    ├── __init__.py
    ├── embeddings.py
    ├── vector_db.py
    ├── bm25_index.py
    ├── reranker.py
    └── hybrid_search.py
```

---

## Kaggle Notebook Setup

### **Шаг 1: Создать Notebook**

Kaggle → Code → New Notebook

**Settings:**
- ✅ **Accelerator: GPU T4** (или P100)
- ✅ **Internet: ON**
- ✅ **Add Dataset**: rag-indices (ваш dataset с индексами)
- ✅ **Add Dataset**: вопросы + сайты (если нужно)

### **Шаг 2: Установить библиотеки**

```python
# Cell 1: Установка
!pip install -q sentence-transformers chromadb rank-bm25 loguru langchain-text-splitters tqdm
```

### **Шаг 3: Скопировать код**

Создайте структуру папок и скопируйте файлы:

```python
# Cell 2: Создание структуры
!mkdir -p src/rag src/preprocessing scripts submissions
```

Затем создайте ячейки с кодом (используйте `%%writefile`):

```python
# Cell 3: Config
%%writefile src/config.py
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openrouter_api_key: Optional[str] = None
    vector_db_path: str = "/kaggle/input/rag-indices/vectordb"  # ВАЖНО: путь к dataset!
    model_name: str = "sergeyzh/rubert-mini-frida"
    embedding_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

```python
# Cell 4-10: Остальные файлы src/
# (data_loader.py, embeddings.py, vector_db.py, bm25_index.py, 
#  reranker.py, hybrid_search.py, preprocessing files)
```

**ИЛИ** загрузите весь `src/` как zip и распакуйте:

```python
# Если загрузили src.zip
!unzip -q /kaggle/input/your-code-dataset/src.zip -d /kaggle/working/
```

### **Шаг 4: Скрипт генерации**

```python
# Cell N: Генерация submission С reranker
import sys
sys.path.insert(0, '/kaggle/working')

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data_loader import DataLoader
from src.rag.bm25_index import BM25Index
from src.rag.hybrid_search import HybridSearch
from src.rag.reranker import Reranker
from src.rag.vector_db import VectorDatabase

logger.info("Загружаем данные...")
data_loader = DataLoader(data_dir="/kaggle/input/your-data-dataset")
questions_df = data_loader.load_questions()

logger.info("Загружаем Vector DB...")
vector_db = VectorDatabase(collection_name="websites")

logger.info("Загружаем BM25 индекс...")
bm25_index = BM25Index(index_path="/kaggle/input/rag-indices/bm25_index.pkl")
bm25_index.load()

logger.info("Загружаем Reranker (GPU!)...")
reranker = Reranker()  # Это быстро на GPU!

logger.info("Инициализируем Hybrid Search...")
hybrid_search = HybridSearch(
    vector_db=vector_db,
    bm25_index=bm25_index,
    reranker=reranker,
    use_reranker=True,
)

# Генерация
results = []
for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
    q_id = int(row["q_id"])
    query = str(row["query"])
    
    top_web_ids = hybrid_search.search(
        query=query,
        bm25_top_k=50,
        vector_top_k=50,
        rerank_top_k=20,
        final_top_k=5,
        use_reranker=True,
    )
    
    results.append({"q_id": q_id, "web_list": top_web_ids})

# Сохранение
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)

logger.info("Готово! submission.csv создан")
```

---

## Альтернатива: Простой способ (zip всего)

### **На локальной машине:**

```powershell
# Создать архив с кодом
Compress-Archive -Path src -DestinationPath src.zip

# Создать архив с индексами  
Compress-Archive -Path artifacts\vectordb,artifacts\bm25_index.pkl -DestinationPath indices.zip
```

### **На Kaggle:**

1. Загрузить `src.zip` как Dataset
2. Загрузить `indices.zip` как Dataset
3. В Notebook:

```python
!unzip -q /kaggle/input/src-dataset/src.zip
!unzip -q /kaggle/input/indices-dataset/indices.zip -d /kaggle/working/artifacts/
```

---

## Преимущества Kaggle GPU

**Reranker на GPU:**
- ❌ **Локально (CPU)**: 23 часа
- ✅ **Kaggle (GPU)**: ~10-15 минут

**Cross-Encoder** в 100-200 раз быстрее на GPU!

---

## Проверка перед запуском

```python
# Cell: Проверка
import os

# Проверка индексов
assert os.path.exists("/kaggle/input/rag-indices/vectordb"), "Vector DB не найдена!"
assert os.path.exists("/kaggle/input/rag-indices/bm25_index.pkl"), "BM25 не найден!"

# Проверка данных
assert os.path.exists("/kaggle/input/your-data/questions_clean.csv"), "Questions не найдены!"

# Проверка GPU
!nvidia-smi

print("✅ Всё готово к запуску!")
```

---

## Полезные советы

1. **Commit & Run** вместо **Save Version** - быстрее
2. **T4 GPU** достаточно (P100 быстрее, но реже доступна)
3. **Логирование** отключите для скорости:
   ```python
   import logging
   logging.getLogger("transformers").setLevel(logging.ERROR)
   ```

4. **Batch reranking** (если медленно):
   ```python
   # В reranker.py измените batch_size
   scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
   ```

---

## Что НЕ нужно выгружать

- ❌ `venv/` (виртуальное окружение)
- ❌ `submissions/` (старые submission)
- ❌ `.git/` 
- ❌ Ноутбуки из `notebooks/`

---

## Итоговый размер

- **Индексы**: ~550MB
- **Код**: ~50KB
- **Данные**: ~10MB (если нужно)

**Total**: ~560MB (без проблем для Kaggle)

---

## Быстрый чеклист

- [ ] Создать Dataset с `vectordb/` и `bm25_index.pkl`
- [ ] Создать Notebook с GPU T4
- [ ] Добавить Dataset к Notebook
- [ ] Установить библиотеки
- [ ] Скопировать `src/` код
- [ ] Исправить пути в `config.py`
- [ ] Запустить генерацию
- [ ] Скачать `submission.csv`

---

Готово! 🚀 На Kaggle GPU это будет ~15 минут вместо 23 часов!

