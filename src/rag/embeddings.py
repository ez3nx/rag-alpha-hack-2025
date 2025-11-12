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
        self.supports_prompts = False
        self._load_model()

    def _load_model(self):
        """Загружает модель эмбеддингов"""
        try:
            self.model = SentenceTransformer(self.model_name)
            
            # Определяем, поддерживает ли модель префиксы (prompts)
            self.supports_prompts = self._check_prompt_support()
            
            logger.info(f"Модель {self.model_name} загружена")
            if self.supports_prompts:
                logger.info(f"✓ Модель поддерживает префиксы (prompts)")
            else:
                logger.info(f"✗ Модель НЕ поддерживает префиксы (используется базовый encode)")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def _check_prompt_support(self) -> bool:
        """
        Проверяет, поддерживает ли модель префиксы (prompts)
        
        Модели с поддержкой:
        - rubert-mini-frida
        - FRIDA
        - intfloat/multilingual-e5-* (с 'query: ' и 'passage: ')
        """
        # Список моделей с поддержкой prefixes/prompts
        prompt_models = [
            "rubert-mini-frida",
            "frida",
            "intfloat/e5",
            "intfloat/multilingual-e5",
        ]
        
        model_lower = self.model_name.lower()
        
        # Проверяем по имени модели
        for prompt_model in prompt_models:
            if prompt_model in model_lower:
                return True
        
        # Дополнительная проверка: есть ли prompts в config модели
        if hasattr(self.model, '_first_module'):
            if hasattr(self.model._first_module(), 'prompts'):
                return True
        
        return False

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
                   Используется только если модель поддерживает prefixes
            show_progress_bar: Показывать ли прогресс бар
        """
        if not self.model:
            raise ValueError("Модель не загружена")

        # Используем prompt только если модель поддерживает И передан prefix
        if self.supports_prompts and prefix:
            try:
                return self.model.encode(
                    texts, 
                    prompt=prefix,
                    convert_to_tensor=False,
                    show_progress_bar=show_progress_bar,
                )
            except Exception as e:
                # Fallback: если что-то пошло не так с prompt
                logger.warning(f"Ошибка при использовании prefix '{prefix}': {e}. Используем базовый encode.")
                return self.model.encode(
                    texts, 
                    convert_to_tensor=False,
                    show_progress_bar=show_progress_bar,
                )
        else:
            # Базовый encode для моделей без поддержки prefixes
            return self.model.encode(
                texts, 
                convert_to_tensor=False,
                show_progress_bar=show_progress_bar,
            )

    def encode_single(self, text: str, prefix: Optional[str] = None) -> np.ndarray:
        """Создает эмбеддинг для одного текста"""
        return self.encode([text], prefix=prefix)[0]