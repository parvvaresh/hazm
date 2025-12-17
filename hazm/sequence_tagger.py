"""این ماژول شامل کلاس‌ها و توابعی برای برچسب‌گذاری توکن‌هاست."""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pycrfsuite import Tagger
from pycrfsuite import Trainer
from sklearn.metrics import accuracy_score

from hazm.types import ChunkedSentence
from hazm.types import Sentence
from hazm.types import TaggedSentence
from hazm.types import Token

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


def data_maker(tokens: list[Sentence]) -> list[list[dict[str, Any]]]:
    """تابع دیتا میکر پیش‌فرض."""
    return [[features(sent, index) for index in range(len(sent))] for sent in tokens]


def iob_features(words: list[str], pos_tags: list[str], index: int) -> dict[str, Any]:
    """ویژگی‌های IOB را برمی‌گرداند (شامل تگ‌های POS)."""
    word_features = features(words, index)
    word_features.update(
        {
            "pos": pos_tags[index],
            "prev_pos": "" if index == 0 else pos_tags[index - 1],
            "next_pos": "" if index == len(pos_tags) - 1 else pos_tags[index + 1],
        },
    )
    return word_features


def iob_data_maker(tokens: list[TaggedSentence]) -> list[list[dict[str, Any]]]:
    """تابع دیتا میکر مخصوص IOB (چون ورودی شامل POS Tag است)."""
    words = [[word for word, _ in token] for token in tokens]
    tags = [[tag for _, tag in token] for token in tokens]
    return [
        [
            iob_features(words=word_tokens, pos_tags=tag_tokens, index=index)
            for index in range(len(word_tokens))
        ]
        for word_tokens, tag_tokens in zip(words, tags, strict=False)
    ]


class SequenceTagger:
    """کلاس پایه برای برچسب‌گذاری توکن‌ها با استفاده از CRFSuite."""

    def __init__(
        self,
        model: str | Path | None = None,
        data_maker: Callable = data_maker,
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

    def tag(self, tokens: Sentence) -> TaggedSentence:
        """یک جمله را برچسب‌گذاری می‌کند."""
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)

        features_list = self.data_maker([tokens])[0]
        tags = self.model.tag(features_list)

        return list(zip(tokens, tags, strict=False))

    def tag_sents(self, sentences: list[Sentence]) -> list[TaggedSentence]:
        """لیستی از جملات را برچسب‌گذاری می‌کند."""
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)

        features_lists = self.data_maker(sentences)
        results = []
        for tokens, feats in zip(sentences, features_lists, strict=False):
            tags = self.model.tag(feats)
            results.append(list(zip(tokens, tags, strict=False)))
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

        inputs = [[x for x, _ in sent] for sent in tagged_list]
        labels = [[y for _, y in sent] for sent in tagged_list]
        features_data = self.data_maker(inputs)

        for xseq, yseq in zip(features_data, labels, strict=False):
            trainer.append(xseq, yseq)

        start_time = time.time()
        trainer.train(file_name)
        end_time = time.time()

        if report_duration:
            logger.info("Training time: %.2f sec", end_time - start_time)

        self.load_model(file_name)

    def save_model(self, filename: str) -> None:
        """مدل را ذخیره می‌کند."""
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)
        self.model.dump(filename)

    def evaluate(self, tagged_sent: list[TaggedSentence]) -> float:
        """ارزیابی مدل."""
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)

        inputs = [[x for x, _ in sent] for sent in tagged_sent]
        gold_labels = [y for sent in tagged_sent for _, y in sent]

        predicted_sents = self.tag_sents(inputs)
        predicted_labels = [tag for sent in predicted_sents for _, tag in sent]

        return float(accuracy_score(gold_labels, predicted_labels))


class IOBTagger(SequenceTagger):
    """کلاس IOBTagger برای تقطیع متن (Chunking)."""

    def __init__(
        self,
        model: str | Path | None = None,
        data_maker: Callable = iob_data_maker,
    ) -> None:
        super().__init__(model, data_maker)

    def __iob_format(
        self,
        tagged_data: TaggedSentence,
        chunk_tags: TaggedSentence,
    ) -> ChunkedSentence:
        """فرمت خروجی را به صورت (word, pos, chunk) در می‌آورد."""
        return [
            (token[0], token[1], chunk_tag[1])
            for token, chunk_tag in zip(tagged_data, chunk_tags, strict=False)
        ]

    def tag(self, tagged_data: TaggedSentence) -> ChunkedSentence:
        """یک جمله را برچسب‌گذاری IOB می‌کند."""
        chunk_tags = super().tag(tagged_data)
        return self.__iob_format(tagged_data, chunk_tags)

    def tag_sents(self, sentences: list[TaggedSentence]) -> list[ChunkedSentence]:
        """لیستی از جملات را برچسب‌گذاری می‌کند."""
        chunk_tags_list = super().tag_sents(sentences)
        return [
            self.__iob_format(tagged_data, chunks)
            for tagged_data, chunks in zip(sentences, chunk_tags_list, strict=False)
        ]

    def train(
        self,
        tagged_list: list[ChunkedSentence],
        c1: float = 0.4,
        c2: float = 0.04,
        max_iteration: int = 400,
        verbose: bool = True,
        file_name: str = "crf.model",
        report_duration: bool = True,
    ) -> None:
        """مدل را آموزش می‌دهد."""
        compatible_tagged_list = [
            [((word, tag), chunk) for word, tag, chunk in sent]
            for sent in tagged_list
        ]

        return super().train(
            compatible_tagged_list,
            c1,
            c2,
            max_iteration,
            verbose,
            file_name,
            report_duration,
        )

    def evaluate(self, tagged_sent: list[ChunkedSentence]) -> float:
        """ارزیابی مدل."""
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)

        inputs = [[(word, tag) for word, tag, _ in sent] for sent in tagged_sent]
        gold_labels = [chunk for sent in tagged_sent for _, _, chunk in sent]

        predicted_sents = self.tag_sents(inputs)
        predicted_labels = [chunk for sent in predicted_sents for _, _, chunk in sent]

        return float(accuracy_score(gold_labels, predicted_labels))
