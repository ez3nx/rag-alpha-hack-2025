"""
Скрипт для подготовки файлов для загрузки на Kaggle
"""
import shutil
from pathlib import Path

from loguru import logger


def main():
    logger.info("Подготовка файлов для Kaggle...")
    
    # Создаем папку для экспорта
    export_dir = Path("kaggle_export")
    export_dir.mkdir(exist_ok=True)
    
    # 1. Копируем индексы
    logger.info("Копируем индексы...")
    indices_dir = export_dir / "indices"
    indices_dir.mkdir(exist_ok=True)
    
    # Vector DB
    if Path("artifacts/vectordb").exists():
        shutil.copytree("artifacts/vectordb", indices_dir / "vectordb", dirs_exist_ok=True)
        logger.info("✓ vectordb скопирован")
    else:
        logger.warning("! vectordb не найден")
    
    # BM25
    if Path("artifacts/bm25_index.pkl").exists():
        shutil.copy("artifacts/bm25_index.pkl", indices_dir / "bm25_index.pkl")
        logger.info("✓ bm25_index.pkl скопирован")
    else:
        logger.warning("! bm25_index.pkl не найден")
    
    # 2. Копируем код
    logger.info("Копируем код...")
    src_dir = export_dir / "src"
    if Path("src").exists():
        shutil.copytree("src", src_dir, dirs_exist_ok=True)
        logger.info("✓ src/ скопирован")
    
    # 3. Создаем README
    readme = export_dir / "README.md"
    readme.write_text("""
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
""")
    
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Готово! Файлы в папке: {export_dir}")
    logger.info(f"{'=' * 50}")
    logger.info("\nСледующие шаги:")
    logger.info("1. Создайте архив:")
    logger.info(f"   Compress-Archive -Path {export_dir}\\indices -DestinationPath kaggle_indices.zip")
    logger.info("2. Загрузите kaggle_indices.zip как Kaggle Dataset")
    logger.info("3. См. KAGGLE_SETUP.md для деталей")
    

if __name__ == "__main__":
    main()

