from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.preprocessing.base import BaseChunker


class NoChunker(BaseChunker):
    """Стратегия без чанкинга - использует документ целиком"""

    def chunk_document(self, document: Dict) -> List[Dict]:
        """Возвращает документ целиком без разбиения"""
        # Комбинируем title + text для поиска
        content = f"{document['title']}\n\n{document['text']}"
        
        return [
            {
                "content": content,
                "metadata": {
                    "web_id": document["web_id"],
                    "url": document["url"],
                    "kind": document["kind"],
                    "title": document["title"],
                    "chunk_id": f"web_{document['web_id']}",
                },
            }
        ]

    def get_strategy_name(self) -> str:
        return "no_chunking"


class RecursiveChunker(BaseChunker):
    """Стратегия с рекурсивным разбиением текста"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "],
        )

    def chunk_document(self, document: Dict) -> List[Dict]:
        """Разбивает документ на чанки"""
        # Комбинируем title + text
        full_text = f"{document['title']}\n\n{document['text']}"
        chunks = self.splitter.split_text(full_text)

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                {
                    "content": chunk,
                    "metadata": {
                        "web_id": document["web_id"],
                        "url": document["url"],
                        "kind": document["kind"],
                        "title": document["title"],
                        "chunk_id": f"web_{document['web_id']}_chunk_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                }
            )

        return chunked_docs

    def get_strategy_name(self) -> str:
        return f"recursive_{self.chunk_size}_{self.chunk_overlap}"


class TitleAwareChunker(BaseChunker):
    """Стратегия, которая всегда включает title в каждый чанк"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "],
        )

    def chunk_document(self, document: Dict) -> List[Dict]:
        """Разбивает текст на чанки, добавляя title к каждому"""
        title = document["title"]
        text = document["text"]
        
        # Разбиваем только text
        text_chunks = self.splitter.split_text(text)

        chunked_docs = []
        for i, chunk in enumerate(text_chunks):
            # Добавляем title к каждому чанку для лучшего контекста
            content = f"Заголовок: {title}\n\n{chunk}"
            
            chunked_docs.append(
                {
                    "content": content,
                    "metadata": {
                        "web_id": document["web_id"],
                        "url": document["url"],
                        "kind": document["kind"],
                        "title": title,
                        "chunk_id": f"web_{document['web_id']}_chunk_{i}",
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                }
            )

        return chunked_docs

    def get_strategy_name(self) -> str:
        return f"title_aware_{self.chunk_size}_{self.chunk_overlap}"