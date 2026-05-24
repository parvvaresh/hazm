"""This module contains classes and functions for sentence tokenization."""


import re

from nltk.tokenize.api import TokenizerI


class SentenceTokenizer(TokenizerI):
    """This class includes functions for extracting sentences from text."""

    def __init__(self: "SentenceTokenizer") -> None:
        """Constructor."""
        self.pattern = re.compile(r"([!.?⸮؟]+)[ \n]+")

    def tokenize(self: "SentenceTokenizer", text: str) -> list[str]:
        """Tokenizes the text into sentences.

        Examples:
            >>> tokenizer = SentenceTokenizer()
            >>> tokenizer.tokenize('جدا کردن ساده است. تقریبا البته!')
            ['جدا کردن ساده است.', 'تقریبا البته!']

        Args:
            text: The text to be tokenized.

        Returns:
            A list of sentences.
        """
        text = self.pattern.sub(r"\1\n\n", text)
        return [
            sentence.replace("\n", " ").strip()
            for sentence in text.split("\n\n")
            if sentence.strip()
        ]


def sent_tokenize(text: str) -> list[str]:
    """Tokenizes text into sentences.

    Args:
        text: The text to tokenize.

    Returns:
        A list of sentences.
    """
    return SentenceTokenizer().tokenize(text)
