from abc import ABC, abstractmethod
from typing import Dict, List


class BaseChunker(ABC):
    """Базовый класс для всех стратегий чанкинга"""

    @abstractmethod
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Разбивает документ на чанки
        
        Args:
            document: Документ с полями:
                - web_id: int
                - title: str
                - text: str
                - url: str
                - kind: str
        
        Returns:
            Список чанков, каждый с полями:
                - content: str (текст чанка)
                - metadata: dict (метаданные включая web_id, chunk_id и т.д.)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Возвращает название стратегии"""
        pass

