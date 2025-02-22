import re
from typing import List

import pandas as pd
from datasets import Dataset, load_dataset

from .config import set_logger

logger = set_logger()


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def split_into_sentences(self, text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]

    def load_and_prepare_data(self) -> Dataset:
        raw_dataset = load_dataset(self.config.id, split="train")
        df = raw_dataset.to_pandas().sample(frac=1).reset_index(drop=True)
        max_news = min(self.config.max_news or len(df), len(df))
        logger.info(f"Number of news: {max_news}")
        sample_df = df.sample(max_news)

        combined_texts = sample_df["text"].to_list() + sample_df["title"].to_list()
        sentences = [
            phrase
            for text in combined_texts
            if text
            for phrase in self.split_into_sentences(text)
        ]
        max_sentences = min(self.config.max_sentences or len(sentences), len(sentences))
        logger.info(f"Number of sentences: {max_sentences}")
        sentences_sample = pd.Series(sentences).sample(max_sentences).to_list()
        return Dataset.from_dict({"text": sentences_sample})
