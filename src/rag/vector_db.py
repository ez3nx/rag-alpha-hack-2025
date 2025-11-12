from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from loguru import logger

from src.config import settings
from src.rag.embeddings import EmbeddingModel


class VectorDatabase:
    def __init__(self, collection_name: str = "websites"):
        try:
            self.client = chromadb.PersistentClient(
                path=settings.vector_db_path, settings=Settings(allow_reset=True)
            )
            self.collection_name = collection_name
            self.collection = None
            self.embedding_model = EmbeddingModel()
            self._setup_collection()
        except Exception as e:
            logger.error(f"Ошибка инициализации VectorDatabase: {e}")
            raise

    def _setup_collection(self):
        """Инициализирует коллекцию"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Коллекция {self.collection_name} готова")
        except Exception as e:
            logger.error(f"Ошибка создания коллекции: {e}")
            raise

    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Добавляет документы в базу батчами"""
        if not documents:
            return

        total_docs = len(documents)
        logger.info(f"Начинаем добавление {total_docs} документов батчами по {batch_size}...")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            texts = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            
            # ID берем из метаданных (web_id или chunk_id)
            ids = [str(doc["metadata"].get("chunk_id", doc["metadata"]["web_id"])) for doc in batch]

            # Создаем эмбеддинги с префиксом для документов
            logger.info(f"Создаем эмбеддинги для батча {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}...")
            embeddings = self.embedding_model.encode(
                texts, 
                prefix=settings.embedding_prefix,
                show_progress_bar=False,
            )

            # Добавляем в базу
            self.collection.add(  # type: ignore
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

        logger.info(f"Добавлено {total_docs} документов в базу")

    def search(
        self, query: str, n_results: int = 5, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Поиск по запросу, возвращает топ-N результатов"""
        query_embedding = self.embedding_model.encode_single(
            query, 
            prefix=settings.query_prefix
        )

        where_filter = {}
        if filters:
            for key, value in filters.items():
                if value is not None:
                    where_filter[key] = value

        results = self.collection.query(  # type: ignore
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter if where_filter else None,
        )

        formatted_results = []
        for i in range(len(results["documents"][0])):  # type: ignore
            formatted_results.append(
                {
                    "content": results["documents"][0][i],  # type: ignore
                    "metadata": results["metadatas"][0][i],  # type: ignore
                    "distance": results["distances"][0][i],  # type: ignore
                    "web_id": results["metadatas"][0][i].get("web_id"),  # type: ignore
                }
            )

        return formatted_results

    def get_collection_info(self) -> Dict:
        """Информация о коллекции"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),  # type: ignore
        }
