"""
Cross-Encoder Reranker для улучшения качества ранжирования
"""
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-Encoder для reranking результатов поиска"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ):
        """
        Args:
            model_name: Название модели Cross-Encoder
                - BAAI/bge-reranker-base (multilingual, good for Russian)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (English)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Загружает Cross-Encoder модель"""
        try:
            logger.info(f"Загружаем Cross-Encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info(f"Cross-Encoder {self.model_name} загружен")
        except Exception as e:
            logger.error(f"Ошибка загрузки Cross-Encoder: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank документов с помощью Cross-Encoder
        
        Args:
            query: Поисковый запрос
            documents: Список документов с 'content' и 'metadata'
            top_k: Количество топовых результатов для возврата
            
        Returns:
            Отранжированные документы с добавленным полем 'rerank_score'
        """
        if not documents:
            return []

        if not self.model:
            raise ValueError("Cross-Encoder модель не загружена")

        # Подготавливаем пары (query, document)
        pairs = [[query, doc["content"]] for doc in documents]

        # Получаем scores от Cross-Encoder
        logger.info(f"Reranking {len(documents)} документов...")
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Добавляем scores к документам
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Сортируем по rerank_score (descending)
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

