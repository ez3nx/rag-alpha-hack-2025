"""
Hybrid Search: комбинация BM25 и Vector Search с Reciprocal Rank Fusion
"""
from collections import defaultdict
from typing import Dict, List, Optional

from loguru import logger

from src.rag.bm25_index import BM25Index
from src.rag.reranker import Reranker
from src.rag.vector_db import VectorDatabase


class HybridSearch:
    """Гибридный поиск: BM25 + Vector Search + RRF + Reranking"""

    def __init__(
        self,
        vector_db: VectorDatabase,
        bm25_index: BM25Index,
        reranker: Optional[Reranker] = None,
        use_reranker: bool = True,
    ):
        self.vector_db = vector_db
        self.bm25_index = bm25_index
        self.reranker = reranker if reranker else (Reranker() if use_reranker else None)

    def reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict]],
        k: int = 60,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)
        
        Формула: RRF(d) = sum(1 / (k + rank(d)))
        
        Args:
            results_list: Список результатов от разных методов поиска
            k: Константа для RRF (обычно 60)
            
        Returns:
            Объединенные и отранжированные результаты
        """
        # Словарь для хранения RRF scores
        rrf_scores = defaultdict(float)
        # Словарь для хранения документов (по chunk_id или web_id)
        doc_map = {}

        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                # Используем chunk_id как уникальный идентификатор
                doc_id = doc["metadata"].get("chunk_id", f"web_{doc['metadata']['web_id']}")
                
                # RRF score
                rrf_scores[doc_id] += 1.0 / (k + rank)
                
                # Сохраняем документ
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Сортируем по RRF score
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Формируем результаты
        results = []
        for doc_id, rrf_score in sorted_docs:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = rrf_score
            results.append(doc)

        return results

    def aggregate_by_web_id(
        self,
        documents: List[Dict],
        top_k: int = 5,
        aggregation_method: str = "max",
    ) -> List[int]:
        """
        Агрегирует чанки по web_id
        
        Args:
            documents: Список документов с scores
            top_k: Количество топовых web_id для возврата
            aggregation_method: Метод агрегации ('max', 'mean', 'min')
            
        Returns:
            Список топ-K web_id
        """
        # Группируем по web_id
        web_id_scores = defaultdict(list)
        
        for doc in documents:
            web_id = doc["metadata"]["web_id"]
            # Используем rerank_score если есть, иначе rrf_score
            score = doc.get("rerank_score", doc.get("rrf_score", 0))
            web_id_scores[web_id].append(score)

        # Агрегируем scores
        web_id_final_scores = {}
        for web_id, scores in web_id_scores.items():
            if aggregation_method == "max":
                web_id_final_scores[web_id] = max(scores)
            elif aggregation_method == "mean":
                web_id_final_scores[web_id] = sum(scores) / len(scores)
            elif aggregation_method == "min":
                web_id_final_scores[web_id] = min(scores)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Сортируем и берем топ-K
        sorted_web_ids = sorted(
            web_id_final_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [web_id for web_id, _ in sorted_web_ids[:top_k]]

    def search(
        self,
        query: str,
        bm25_top_k: int = 50,
        vector_top_k: int = 50,
        rerank_top_k: int = 20,
        final_top_k: int = 5,
        use_reranker: bool = True,
    ) -> List[int]:
        """
        Гибридный поиск
        
        Pipeline:
        1. BM25 поиск (top 50)
        2. Vector поиск (top 50)
        3. RRF fusion
        4. Reranking (top 20)
        5. Агрегация по web_id (top 5)
        
        Args:
            query: Поисковый запрос
            bm25_top_k: Количество результатов из BM25
            vector_top_k: Количество результатов из Vector Search
            rerank_top_k: Количество документов для reranking
            final_top_k: Финальное количество web_id
            use_reranker: Использовать ли reranker
            
        Returns:
            Список топ-K web_id
        """
        # 1. BM25 поиск
        logger.info(f"BM25 поиск для: '{query[:50]}...'")
        bm25_results = self.bm25_index.search(query, n_results=bm25_top_k)

        # 2. Vector поиск
        logger.info(f"Vector поиск для: '{query[:50]}...'")
        vector_results = self.vector_db.search(query, n_results=vector_top_k)

        # 3. RRF Fusion
        logger.info("Reciprocal Rank Fusion...")
        fused_results = self.reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=60,
        )

        # 4. Reranking (опционально)
        if use_reranker and self.reranker and len(fused_results) > 0:
            logger.info(f"Reranking топ-{rerank_top_k} документов...")
            reranked = self.reranker.rerank(
                query,
                fused_results[:rerank_top_k],
                top_k=rerank_top_k,
            )
        else:
            reranked = fused_results[:rerank_top_k]

        # 5. Агрегация по web_id
        logger.info(f"Агрегация по web_id (топ-{final_top_k})...")
        top_web_ids = self.aggregate_by_web_id(
            reranked,
            top_k=final_top_k,
            aggregation_method="max",
        )

        return top_web_ids

