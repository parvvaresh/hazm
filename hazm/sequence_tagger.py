"""این ماژول شامل کلاس‌ها و توابعی برای برچسب‌گذاری توکن‌هاست."""

import time
import logging
from typing import Any, Callable, List, Tuple
from pathlib import Path

import numpy as np
from pycrfsuite import Tagger, Trainer
from sklearn.metrics import accuracy_score

from hazm.types import TaggedSentence, Token

logger = logging.getLogger(__name__)


def features(sent: list[Token], index: int) -> dict[str, Any]:
    """فهرست فیچرها را برمی‌گرداند."""
    return {
        "word": sent[index],
        "is_first": index == 0,
        "is_last": index == len(sent) - 1,
        "is_num": sent[index].isdigit(),
        "prev_word": sent[index - 1] if index != 0 else "",
        "next_word": sent[index + 1] if index != len(sent) - 1 else "",
    }


def data_maker(tokens: list[list[Token]]) -> list[list[dict[str, Any]]]:
    """تابع دیتا میکر پیش‌فرض."""
    return [[features(sent, index) for index in range(len(sent))] for sent in tokens]


class SequenceTagger:
    """کلاس پایه برای برچسب‌گذاری توکن‌ها با استفاده از CRFSuite."""

    def __init__(
        self, 
        model: str | Path | None = None, 
        data_maker: Callable = data_maker
    ) -> None:
        self.model: Tagger | None = None
        if model is not None:
            self.load_model(model)
        self.data_maker = data_maker

    def load_model(self, model_path: str | Path) -> None:
        """فایل تگر را بارگزاری می‌کند."""
        tagger = Tagger()
        tagger.open(str(model_path))
        self.model = tagger

    def tag(self, tokens: list[Token]) -> TaggedSentence:
        """یک جمله را برچسب‌گذاری می‌کند."""
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        features_list = self.data_maker([tokens])[0]
        tags = self.model.tag(features_list)
        
        return list(zip(tokens, tags, strict=True))

    def tag_sents(self, sentences: list[list[Token]]) -> list[TaggedSentence]:
        """لیستی از جملات را برچسب‌گذاری می‌کند."""
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        features_lists = self.data_maker(sentences)
        results = []
        for tokens, feats in zip(sentences, features_lists, strict=True):
            tags = self.model.tag(feats)
            results.append(list(zip(tokens, tags, strict=True)))
        return results

    def train(
        self,
        tagged_list: list[TaggedSentence],
        c1: float = 0.4,
        c2: float = 0.04,
        max_iteration: int = 400,
        verbose: bool = True,
        file_name: str = "crf.model",
        report_duration: bool = True,
    ) -> None:
        """مدل را آموزش می‌دهد."""
        trainer = Trainer(verbose=verbose)
        trainer.set_params({
            "c1": c1,
            "c2": c2,
            "max_iterations": max_iteration,
            "feature.possible_transitions": True,
        })

        sentences = [[word for word, _ in sent] for sent in tagged_list]
        labels = [[tag for _, tag in sent] for sent in tagged_list]
        features_data = self.data_maker(sentences)

        for xseq, yseq in zip(features_data, labels, strict=True):
            trainer.append(xseq, yseq)

        start_time = time.time()
        trainer.train(file_name)
        end_time = time.time()

        if report_duration:
            logger.info(f"Training time: {(end_time - start_time):.2f} sec")

        self.load_model(file_name)

    def save_model(self, filename: str) -> None:
        """مدل را ذخیره می‌کند."""
        pass 

    def evaluate(self, tagged_sent: list[TaggedSentence]) -> float:
        """ارزیابی مدل."""
        if self.model is None:
            raise ValueError("Model is not loaded.")

        tokens = [[word for word, _ in sent] for sent in tagged_sent]
        gold_labels = [tag for sent in tagged_sent for _, tag in sent]
        
        predicted_sents = self.tag_sents(tokens)
        predicted_labels = [tag for sent in predicted_sents for _, tag in sent]
        
        return float(accuracy_score(gold_labels, predicted_labels))