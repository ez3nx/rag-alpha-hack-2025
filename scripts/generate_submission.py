"""
Скрипт для генерации submission файла с топ-5 документами для каждого вопроса
Использует Hybrid Search: BM25 + Vector Search + RRF + Reranking
"""
import ast
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data_loader import DataLoader
from src.rag.bm25_index import BM25Index
from src.rag.hybrid_search import HybridSearch
from src.rag.reranker import Reranker
from src.rag.vector_db import VectorDatabase


def generate_submission(
    output_filename: str = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    bm25_top_k: int = 50,
    vector_top_k: int = 50,
    rerank_top_k: int = 20,
    final_top_k: int = 5,
):
    """
    Генерирует submission файл с Hybrid Search
    
    Args:
        output_filename: Имя выходного файла (если None, генерируется автоматически)
        use_hybrid: Использовать ли hybrid search (BM25 + Vector + RRF)
        use_reranker: Использовать ли reranker (Cross-Encoder)
        bm25_top_k: Количество результатов из BM25
        vector_top_k: Количество результатов из Vector Search
        rerank_top_k: Количество документов для reranking
        final_top_k: Финальное количество web_id (обычно 5)
    """
    logger.info("=" * 50)
    logger.info("Генерация submission файла")
    logger.info(f"Hybrid Search: {use_hybrid}, Reranker: {use_reranker}")
    logger.info("=" * 50)

    # 1. Загружаем данные
    data_loader = DataLoader()
    questions_df = data_loader.load_questions()
    logger.info(f"Загружено {len(questions_df)} вопросов")

    # 2. Инициализируем индексы
    vector_db = VectorDatabase(collection_name="websites")
    info = vector_db.get_collection_info()
    logger.info(f"Подключено к Vector DB: {info}")

    bm25_index = BM25Index()
    if not bm25_index.load():
        logger.error("BM25 индекс не найден! Запустите build_index.py")
        return

    # 3. Инициализируем hybrid search
    if use_hybrid:
        logger.info("Инициализируем Hybrid Search...")
        reranker = Reranker() if use_reranker else None
        hybrid_search = HybridSearch(
            vector_db=vector_db,
            bm25_index=bm25_index,
            reranker=reranker,
            use_reranker=use_reranker,
        )

    # 4. Обрабатываем каждый вопрос
    results = []
    
    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Поиск"):
        q_id = int(row["q_id"])
        query = str(row["query"])

        if use_hybrid:
            # Hybrid Search
            top_web_ids = hybrid_search.search(
                query=query,
                bm25_top_k=bm25_top_k,
                vector_top_k=vector_top_k,
                rerank_top_k=rerank_top_k,
                final_top_k=final_top_k,
                use_reranker=use_reranker,
            )
        else:
            # Обычный vector search (fallback)
            search_results = vector_db.search(query, n_results=vector_top_k)
            
            # Агрегируем по web_id
            web_id_scores = {}
            for result in search_results:
                web_id = result["web_id"]
                distance = result["distance"]
                
                if web_id not in web_id_scores:
                    web_id_scores[web_id] = distance
                else:
                    web_id_scores[web_id] = min(web_id_scores[web_id], distance)

            sorted_web_ids = sorted(web_id_scores.items(), key=lambda x: x[1])
            top_web_ids = [web_id for web_id, _ in sorted_web_ids[:final_top_k]]

        results.append({"q_id": q_id, "web_list": top_web_ids})

    # 4. Создаем DataFrame
    submission_df = pd.DataFrame(results)
    
    # 5. Сохраняем
    output_dir = Path("submissions")
    output_dir.mkdir(exist_ok=True)
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"submission_{timestamp}.csv"
    
    output_path = output_dir / output_filename
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission сохранен в: {output_path}")
    logger.info(f"Всего вопросов: {len(submission_df)}")
    
    # 6. Статистика
    logger.info("\n" + "=" * 50)
    logger.info("Статистика submission:")
    logger.info("=" * 50)
    
    # Проверяем, что все вопросы имеют ровно top_k результатов
    web_list_lengths = submission_df["web_list"].apply(len)
    logger.info(f"Все вопросы имеют {top_k} результатов: {(web_list_lengths == top_k).all()}")
    
    # Топ упоминаемых web_id
    all_web_ids = []
    for web_list in submission_df["web_list"]:
        all_web_ids.extend(web_list)
    
    top_mentioned = Counter(all_web_ids).most_common(10)
    logger.info("\nТоп-10 наиболее упоминаемых web_id:")
    for web_id, count in top_mentioned:
        logger.info(f"  web_id {web_id}: {count} раз")
    
    logger.info("=" * 50)
    logger.info("Готово!")
    logger.info("=" * 50)

    return output_path


if __name__ == "__main__":
    import sys

    # Можно передать имя файла как аргумент
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    generate_submission(output_filename=filename)

