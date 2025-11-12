from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""
    
    openrouter_api_key: Optional[str] = None
    vector_db_path: str = "artifacts/vectordb"
    model_name: str = "sergeyzh/rubert-mini-frida"
    embedding_prefix: str = "search_document: "  # Для индексирования документов
    query_prefix: str = "search_query: "  # Для поисковых запросов
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
