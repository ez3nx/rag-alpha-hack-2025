"""
Быстрая генерация submission БЕЗ reranker (только BM25 + Vector + RRF)
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import Counter
from datetime import datetime

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data_loader import DataLoader
from src.rag.bm25_index import BM25Index
from src.rag.hybrid_search import HybridSearch
from src.rag.vector_db import VectorDatabase


def main():
    logger.info("=" * 50)
    logger.info("Генерация submission (БЕЗ reranker - быстро!)")
    logger.info("=" * 50)

    # 1. Загружаем данные
    data_loader = DataLoader()
    questions_df = data_loader.load_questions()
    logger.info(f"Загружено {len(questions_df)} вопросов")

    # 2. Инициализируем индексы
    logger.info("Загружаем Vector DB...")
    vector_db = VectorDatabase(collection_name="websites")
    info = vector_db.get_collection_info()
    logger.info(f"Vector DB: {info}")

    logger.info("Загружаем BM25 индекс...")
    bm25_index = BM25Index()
    if not bm25_index.load():
        logger.error("BM25 индекс не найден! Запустите build_index.py")
        return

    # 3. Инициализируем hybrid search БЕЗ reranker
    logger.info("Инициализируем Hybrid Search (БЕЗ reranker)...")
    hybrid_search = HybridSearch(
        vector_db=vector_db,
        bm25_index=bm25_index,
        reranker=None,
        use_reranker=False,
    )

    # 4. Обрабатываем каждый вопрос
    results = []
    
    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Поиск"):
        q_id = int(row["q_id"])
        query = str(row["query"])

        # Hybrid Search БЕЗ reranker
        top_web_ids = hybrid_search.search(
            query=query,
            bm25_top_k=50,
            vector_top_k=50,
            rerank_top_k=20,  # Не используется без reranker
            final_top_k=5,
            use_reranker=False,
        )

        results.append({"q_id": q_id, "web_list": top_web_ids})

    # 5. Создаем DataFrame
    submission_df = pd.DataFrame(results)
    
    # 6. Сохраняем
    output_dir = Path("submissions")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"hybrid_fast_{timestamp}.csv"
    output_path = output_dir / output_filename
    
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Submission сохранен: {output_path}")
    logger.info(f"Всего вопросов: {len(submission_df)}")
    
    # Статистика
    all_web_ids = []
    for web_list in submission_df["web_list"]:
        all_web_ids.extend(web_list)
    
    top_mentioned = Counter(all_web_ids).most_common(10)
    logger.info("\nТоп-10 наиболее упоминаемых web_id:")
    for web_id, count in top_mentioned:
        logger.info(f"  web_id {web_id}: {count} раз")
    
    logger.info("=" * 50)
    logger.info("Готово!")


if __name__ == "__main__":
    main()

