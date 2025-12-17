from abc import ABC
from abc import abstractmethod

from hazm.types import Sentence
from hazm.types import TaggedSentence
from hazm.types import Token


class NormalizerProtocol(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        """متن را نرمال‌سازی می‌کند."""

class TokenizerProtocol(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[Token]:
        """متن را به توکن‌ها تبدیل می‌کند."""

class LemmatizerProtocol(ABC):
    @abstractmethod
    def lemmatize(self, word: str, pos: str = "") -> str:
        """ریشه کلمه را برمی‌گرداند."""

class TaggerProtocol(ABC):
    @abstractmethod
    def tag(self, tokens: Sentence) -> TaggedSentence:
        """یک جمله را برچسب‌گذاری می‌کند."""

    @abstractmethod
    def tag_sents(self, sentences: list[Sentence]) -> list[TaggedSentence]:
        """لیستی از جملات را برچسب‌گذاری می‌کند."""