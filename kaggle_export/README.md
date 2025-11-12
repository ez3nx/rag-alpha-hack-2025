
# RAG Indices for Kaggle

Содержимое:
- `indices/vectordb/` - ChromaDB векторная база
- `indices/bm25_index.pkl` - BM25 индекс
- `src/` - Исходный код

## Использование на Kaggle

1. Загрузить как Dataset
2. В Notebook добавить этот Dataset
3. Указать пути:
   - vector_db_path: "/kaggle/input/rag-indices/indices/vectordb"
   - bm25_path: "/kaggle/input/rag-indices/indices/bm25_index.pkl"

Подробнее: см. KAGGLE_SETUP.md в репозитории
