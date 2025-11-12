"""
Скрипт для построения векторного индекса и BM25 индекса из данных веб-страниц
"""
from loguru import logger

from src.data_loader import DataLoader
from src.preprocessing import NoChunker
from src.rag.bm25_index import BM25Index
from src.rag.vector_db import VectorDatabase


def build_index(chunker_strategy: str = "no_chunking"):
    """
    Строит векторный индекс
    
    Args:
        chunker_strategy: Стратегия чанкинга ("no_chunking", "recursive", "title_aware")
    """
    logger.info("=" * 50)
    logger.info("Начинаем построение векторного индекса")
    logger.info("=" * 50)

    # 1. Загружаем данные
    data_loader = DataLoader()
    websites = data_loader.load_websites()
    logger.info(f"Загружено {len(websites)} веб-страниц")

    # 2. Выбираем стратегию чанкинга
    if chunker_strategy == "no_chunking":
        from src.preprocessing import NoChunker
        chunker = NoChunker()
    elif chunker_strategy == "recursive":
        from src.preprocessing import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    elif chunker_strategy == "title_aware":
        from src.preprocessing import TitleAwareChunker
        chunker = TitleAwareChunker(chunk_size=800, chunk_overlap=150)
    else:
        raise ValueError(f"Неизвестная стратегия чанкинга: {chunker_strategy}")

    logger.info(f"Используем стратегию чанкинга: {chunker.get_strategy_name()}")

    # 3. Обрабатываем документы
    all_chunks = []
    for i, website in enumerate(websites):
        if i % 100 == 0:
            logger.info(f"Обработано {i}/{len(websites)} документов")
        
        chunks = chunker.chunk_document(website)
        all_chunks.extend(chunks)

    logger.info(f"Создано {len(all_chunks)} чанков из {len(websites)} документов")

    # 4. Создаем векторную базу (ChromaDB)
    logger.info("Создаем векторную базу данных...")
    vector_db = VectorDatabase(collection_name="websites")
    vector_db.add_documents(all_chunks, batch_size=100)

    info = vector_db.get_collection_info()
    logger.info(f"Векторная база данных настроена: {info}")

    # 5. Создаем BM25 индекс
    logger.info("Создаем BM25 индекс...")
    bm25_index = BM25Index()
    bm25_index.build(all_chunks)
    bm25_index.save()

    # 6. Тестовые поиски
    test_query = "Как получить ипотеку?"
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Тестовый поиск по запросу: '{test_query}'")
    logger.info("=" * 50)

    # Vector search
    logger.info("\nVector Search (Top 3):")
    vector_results = vector_db.search(test_query, n_results=3)
    for i, result in enumerate(vector_results, 1):
        logger.info(f"{i}. Web ID: {result['web_id']}, Distance: {result['distance']:.4f}")
        logger.info(f"   Title: {result['metadata']['title'][:80]}...")

    # BM25 search
    logger.info("\nBM25 Search (Top 3):")
    bm25_results = bm25_index.search(test_query, n_results=3)
    for i, result in enumerate(bm25_results, 1):
        logger.info(f"{i}. Web ID: {result['web_id']}, Score: {result['score']:.4f}")
        logger.info(f"   Title: {result['metadata']['title'][:80]}...")

    logger.info("\n" + "=" * 50)
    logger.info("Индексы успешно построены!")
    logger.info("=" * 50)


if __name__ == "__main__":
    import sys

    # По умолчанию используем no_chunking
    strategy = sys.argv[1] if len(sys.argv) > 1 else "no_chunking"
    build_index(chunker_strategy=strategy)

