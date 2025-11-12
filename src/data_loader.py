from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger


class DataLoader:
    """Класс для загрузки данных из CSV файлов"""

    def __init__(self, data_dir: str = "artifacts/data"):
        self.data_dir = Path(data_dir)

    def load_websites(self) -> List[Dict]:
        """Загружает данные веб-страниц"""
        filepath = self.data_dir / "websites.csv"
        logger.info(f"Загружаем данные из {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Загружено {len(df)} веб-страниц")

        # Конвертируем в список словарей
        websites = []
        for _, row in df.iterrows():
            websites.append(
                {
                    "web_id": int(row["web_id"]),
                    "url": str(row["url"]),
                    "kind": str(row["kind"]),
                    "title": str(row["title"]),
                    "text": str(row["text"]),
                }
            )

        return websites

    def load_questions(self) -> pd.DataFrame:
        """Загружает вопросы"""
        filepath = self.data_dir / "questions_clean.csv"
        logger.info(f"Загружаем вопросы из {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Загружено {len(df)} вопросов")

        return df

    def load_sample_submission(self) -> pd.DataFrame:
        """Загружает образец submission"""
        filepath = self.data_dir / "sample_submission.csv"
        logger.info(f"Загружаем sample submission из {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Загружено {len(df)} записей")

        return df

    def get_data_stats(self) -> Dict:
        """Возвращает статистику по данным"""
        websites_df = pd.read_csv(self.data_dir / "websites.csv")
        questions_df = pd.read_csv(self.data_dir / "questions_clean.csv")

        stats = {
            "total_websites": len(websites_df),
            "total_questions": len(questions_df),
            "website_kinds": websites_df["kind"].value_counts().to_dict(),
            "avg_text_length": websites_df["text"].str.len().mean(),
            "max_text_length": websites_df["text"].str.len().max(),
            "min_text_length": websites_df["text"].str.len().min(),
        }

        return stats

