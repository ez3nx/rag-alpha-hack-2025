from typing import List, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or settings.model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Загружает модель эмбеддингов"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Модель {self.model_name} загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def encode(
        self, 
        texts: List[str], 
        prefix: Optional[str] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Создает эмбеддинги для текстов
        
        Args:
            texts: Список текстов
            prefix: Префикс для модели (например, 'search_document: ' или 'search_query: ')
            show_progress_bar: Показывать ли прогресс бар
        """
        if not self.model:
            raise ValueError("Модель не загружена")

        # Используем prompt parameter для rubert-mini-frida
        if prefix:
            return self.model.encode(
                texts, 
                prompt=prefix,
                convert_to_tensor=False,
                show_progress_bar=show_progress_bar,
            )
        else:
            return self.model.encode(
                texts, 
                convert_to_tensor=False,
                show_progress_bar=show_progress_bar,
            )

    def encode_single(self, text: str, prefix: Optional[str] = None) -> np.ndarray:
        """Создает эмбеддинг для одного текста"""
        return self.encode([text], prefix=prefix)[0]