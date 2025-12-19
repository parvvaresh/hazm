from abc import ABC
from abc import abstractmethod

from hazm.types import Sentence
from hazm.types import TaggedSentence
from hazm.types import Token


class NormalizerProtocol(ABC):
    """Protocol for text normalization."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalizes the text.

        Args:
            text: The text to be normalized.

        Returns:
            The normalized text.
        """

class TokenizerProtocol(ABC):
    """Protocol for text tokenization."""

    @abstractmethod
    def tokenize(self, text: str) -> list[Token]:
        """Tokenizes the text into a list of tokens.

        Args:
            text: The text to be tokenized.

        Returns:
            A list of tokens.
        """

class LemmatizerProtocol(ABC):
    """Protocol for lemmatization."""

    @abstractmethod
    def lemmatize(self, word: str, pos: str = "") -> str:
        """Lemmatizes the given word.

        Args:
            word: The word to be lemmatized.
            pos: The part-of-speech tag of the word (optional).

        Returns:
            The lemma of the word.
        """

class TaggerProtocol(ABC):
    """Protocol for part-of-speech tagging."""

    @abstractmethod
    def tag(self, tokens: Sentence) -> TaggedSentence:
        """Tags a single sentence.

        Args:
            tokens: A list of tokens representing a sentence.

        Returns:
            A list of (token, tag) tuples.
        """

    @abstractmethod
    def tag_sents(self, sentences: list[Sentence]) -> list[TaggedSentence]:
        """Tags a list of sentences.

        Args:
            sentences: A list of sentences, where each sentence is a list of tokens.

        Returns:
            A list of tagged sentences.
        """
