"""
BM25 индексер для keyword-based поиска
"""
import pickle
from pathlib import Path
from typing import Dict, List

from loguru import logger
from rank_bm25 import BM25Okapi

from src.config import settings


class BM25Index:
    """BM25 индексер для keyword поиска"""

    def __init__(self, index_path: str = None):
        self.index_path = index_path or str(Path(settings.vector_db_path).parent / "bm25_index.pkl")
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []

    def build(self, documents: List[Dict]):
        """
        Строит BM25 индекс
        
        Args:
            documents: Список документов с полями 'content' и 'metadata'
        """
        logger.info(f"Строим BM25 индекс для {len(documents)} документов...")
        
        self.documents = documents
        
        # Простая токенизация (можно улучшить)
        self.tokenized_corpus = [
            self._tokenize(doc["content"]) 
            for doc in documents
        ]
        
        # Создаем BM25 индекс
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"BM25 индекс построен для {len(self.documents)} документов")

    def _tokenize(self, text: str) -> List[str]:
        """Простая токенизация (lowercase + split)"""
        return text.lower().split()

    def search(self, query: str, n_results: int = 50) -> List[Dict]:
        """
        Ищет документы по BM25
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            
        Returns:
            Список документов с полями 'content', 'metadata', 'score', 'web_id'
        """
        if not self.bm25:
            logger.error("BM25 индекс не построен!")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Сортируем по score (descending)
        top_indices = scores.argsort()[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(scores[idx]),
                "web_id": doc["metadata"]["web_id"],
            })
        
        return results

    def save(self):
        """Сохраняет индекс на диск"""
        logger.info(f"Сохраняем BM25 индекс в {self.index_path}")
        
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
        }
        
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info("BM25 индекс сохранен")

    def load(self):
        """Загружает индекс с диска"""
        if not Path(self.index_path).exists():
            logger.warning(f"BM25 индекс не найден: {self.index_path}")
            return False

        logger.info(f"Загружаем BM25 индекс из {self.index_path}")
        
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]
        
        logger.info(f"BM25 индекс загружен ({len(self.documents)} документов)")
        return True

