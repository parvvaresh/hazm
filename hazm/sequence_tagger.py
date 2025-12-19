"""This module contains classes and functions for tagging tokens."""

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
    """Returns a dictionary of features for the token at the given index in the sentence.

    Args:
        sent: The sentence containing the token.
        index: The index of the token.

    Returns:
        A dictionary of features.
    """
    return {
        "word": sent[index],
        "is_first": index == 0,
        "is_last": index == len(sent) - 1,
        "is_num": sent[index].isdigit(),
        "prev_word": sent[index - 1] if index != 0 else "",
        "next_word": sent[index + 1] if index != len(sent) - 1 else "",
    }


def data_maker(tokens: list[Sentence]) -> list[list[dict[str, Any]]]:
    """Default data maker function.

    Args:
        tokens: A list of sentences.

    Returns:
        A list of lists of feature dictionaries.
    """
    return [[features(sent, index) for index in range(len(sent))] for sent in tokens]


def iob_features(words: list[str], pos_tags: list[str], index: int) -> dict[str, Any]:
    """Returns IOB features (including POS tags).

    Args:
        words: List of words in the sentence.
        pos_tags: List of POS tags for the words.
        index: The index of the word.

    Returns:
        A dictionary of features.
    """
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
    """Data maker function for IOB tagging (includes POS tags).

    Args:
        tokens: A list of tagged sentences.

    Returns:
        A list of lists of feature dictionaries.
    """
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
    """Base class for sequence tagging using CRFSuite.

    Examples:
        >>> tagger = SequenceTagger(model='tagger.model')
        >>> tagger.tag(['من', 'به', 'مدرسه', 'رفتم', '.'])
        [('من', 'PRON'), ('به', 'ADP'), ('مدرسه', 'NOUN'), ('رفتم', 'VERB'), ('.', 'PUNCT')]
    """

    def __init__(
        self,
        model: str | Path | None = None,
        data_maker: Callable = data_maker,
    ) -> None:
        """Constructor.

        Args:
            model: Path to the model file.
            data_maker: Function to generate features from tokens.
        """
        self.model: Tagger | None = None
        if model is not None:
            self.load_model(model)
        self.data_maker = data_maker

    def load_model(self, model_path: str | Path) -> None:
        """Loads the tagger model.

        Examples:
            >>> tagger = SequenceTagger()
            >>> tagger.load_model('tagger.model')

        Args:
            model_path: Path to the model file.
        """
        tagger = Tagger()
        tagger.open(str(model_path))
        self.model = tagger

    def tag(self, tokens: Sentence) -> TaggedSentence:
        """Tags a single sentence.

        Examples:
            >>> tagger = SequenceTagger(model='tagger.model')
            >>> tagger.tag(['من', 'به', 'مدرسه', 'ایران', 'رفته_بودم', '.'])
            [('من', 'PRON'), ('به', 'ADP'), ('مدرسه', 'NOUN,EZ'), ('ایران', 'NOUN'), ('رفته_بودم', 'VERB'), ('.', 'PUNCT')]

        Args:
            tokens: A list of tokens representing a sentence.

        Returns:
            A tagged sentence.
        """
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)

        features_list = self.data_maker([tokens])[0]
        tags = self.model.tag(features_list)

        return list(zip(tokens, tags, strict=False))

    def tag_sents(self, sentences: list[Sentence]) -> list[TaggedSentence]:
        """Tags multiple sentences.

        Examples:
            >>> tagger = SequenceTagger(model='tagger.model')
            >>> tagger.tag_sents([['من', 'به', 'مدرسه', 'ایران', 'رفته_بودم', '.']])
            [[('من', 'PRON'), ('به', 'ADP'), ('مدرسه', 'NOUN,EZ'), ('ایران', 'NOUN'), ('رفته_بودم', 'VERB'), ('.', 'PUNCT')]]

        Args:
            sentences: A list of sentences to tag.

        Returns:
            A list of tagged sentences.
        """
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
        """Trains the model.

        Examples:
            >>> tagger = SequenceTagger()
            >>> tagged_list = [[('من', 'PRON'), ('به', 'ADP'), ('مدرسه', 'NOUN'), ('رفتم', 'VERB'), ('.', 'PUNCT')]]
            >>> tagger.train(tagged_list, c1=0.5, c2=0.5, max_iteration=100, file_name='tagger.model')

        Args:
            tagged_list: A list of tagged sentences for training.
            c1: Coefficient for L1 regularization.
            c2: Coefficient for L2 regularization.
            max_iteration: Maximum number of iterations for training.
            verbose: Whether to print verbose output.
            file_name: The name of the file to save the trained model.
            report_duration: Whether to report the training duration.
        """
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
        """Saves the model to a file.

        Examples:
            >>> tagger.save_model('new_tagger.model')

        Args:
            filename: The name of the file to save the model.
        """
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)
        self.model.dump(filename)

    def evaluate(self, tagged_sent: list[TaggedSentence]) -> float:
        """Evaluates the model.

        Examples:
            >>> tagger = SequenceTagger(model='tagger.model')
            >>> tagger.evaluate([[('من', 'PRON'), ('رفتم', 'VERB')]])
            1.0

        Args:
            tagged_sent: A list of tagged sentences for evaluation.

        Returns:
            The accuracy of the model.
        """
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)

        inputs = [[x for x, _ in sent] for sent in tagged_sent]
        gold_labels = [y for sent in tagged_sent for _, y in sent]

        predicted_sents = self.tag_sents(inputs)
        predicted_labels = [tag for sent in predicted_sents for _, tag in sent]

        return float(accuracy_score(gold_labels, predicted_labels))


class IOBTagger(SequenceTagger):
    """IOB Tagger class for text chunking.

    Examples:
        >>> iob_tagger = IOBTagger(model='chunker.model')
        >>> iob_tagger.tag([('من', 'PRON'), ('به', 'ADP'), ('مدرسه', 'NOUN'), ('رفتم', 'VERB'), ('.', 'PUNCT')])
        [('من', 'PRON', 'B-NP'), ('به', 'ADP', 'B-PP'), ('مدرسه', 'NOUN', 'B-NP'), ('رفتم', 'VERB', 'B-VP'), ('.', 'PUNCT', 'O')]
    """

    def __init__(
        self,
        model: str | Path | None = None,
        data_maker: Callable = iob_data_maker,
    ) -> None:
        """Constructor.

        Args:
            model: Path to the model file.
            data_maker: Function to generate features.
        """
        super().__init__(model, data_maker)

    def __iob_format(
        self,
        tagged_data: TaggedSentence,
        chunk_tags: TaggedSentence,
    ) -> ChunkedSentence:
        """Converts output to (word, pos, chunk) format.

        Args:
            tagged_data: The tagged sentence with POS tags.
            chunk_tags: The chunk tags.

        Returns:
            A chunked sentence.
        """
        return [
            (token[0], token[1], chunk_tag[1])
            for token, chunk_tag in zip(tagged_data, chunk_tags, strict=False)
        ]

    def tag(self, tagged_data: TaggedSentence) -> ChunkedSentence:
        """Tags a single sentence with IOB tags.

        Examples:
            >>> iob_tagger.tag([('من', 'PRON'), ('به', 'ADP'), ('مدرسه', 'NOUN')])
            [('من', 'PRON', 'B-NP'), ('به', 'ADP', 'B-PP'), ('مدرسه', 'NOUN', 'B-NP')]

        Args:
            tagged_data: A tagged sentence.

        Returns:
            A chunked sentence.
        """
        chunk_tags = super().tag(tagged_data)
        return self.__iob_format(tagged_data, chunk_tags)

    def tag_sents(self, sentences: list[TaggedSentence]) -> list[ChunkedSentence]:
        """Tags multiple sentences.

        Examples:
            >>> iob_tagger.tag_sents([[('من', 'PRON'), ('رفتم', 'VERB')]])
            [[('من', 'PRON', 'B-NP'), ('رفتم', 'VERB', 'B-VP')]]

        Args:
            sentences: A list of tagged sentences.

        Returns:
            A list of chunked sentences.
        """
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
        """Trains the model.

        Examples:
            >>> iob_tagger.train(tagged_list=[[('من', 'PRON', 'B-NP'), ('رفتم', 'VERB', 'B-VP')]], file_name='chunker.model')

        Args:
            tagged_list: A list of chunked sentences for training.
            c1: Coefficient for L1 regularization.
            c2: Coefficient for L2 regularization.
            max_iteration: Maximum number of iterations for training.
            verbose: Whether to print verbose output.
            file_name: The name of the file to save the trained model.
            report_duration: Whether to report the training duration.
        """
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
        """Evaluates the model.

        Examples:
            >>> iob_tagger.evaluate([[('من', 'PRON', 'B-NP'), ('رفتم', 'VERB', 'B-VP')]])
            1.0

        Args:
            tagged_sent: A list of chunked sentences for evaluation.

        Returns:
            The accuracy of the model.
        """
        if self.model is None:
            msg = "Model is not loaded."
            raise ValueError(msg)

        inputs = [[(word, tag) for word, tag, _ in sent] for sent in tagged_sent]
        gold_labels = [chunk for sent in tagged_sent for _, _, chunk in sent]

        predicted_sents = self.tag_sents(inputs)
        predicted_labels = [chunk for sent in predicted_sents for _, _, chunk in sent]

        return float(accuracy_score(gold_labels, predicted_labels))
