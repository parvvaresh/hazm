from abc import ABC, abstractmethod
from hazm.types import Sentence, Token, TaggedSentence, Token

class NormalizerProtocol(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        """متن را نرمال‌سازی می‌کند."""
        pass

class TokenizerProtocol(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[Token]:
        """متن را به توکن‌ها تبدیل می‌کند."""
        pass

class LemmatizerProtocol(ABC):
    @abstractmethod
    def lemmatize(self, word: str, pos: str = "") -> str:
        """ریشه کلمه را برمی‌گرداند."""
        pass

class TaggerProtocol(ABC):
    @abstractmethod
    def tag(self, tokens: Sentence) -> TaggedSentence:
        """یک جمله را برچسب‌گذاری می‌کند."""
        pass

    @abstractmethod
    def tag_sents(self, sentences: list[Sentence]) -> list[TaggedSentence]:
        """لیستی از جملات را برچسب‌گذاری می‌کند."""
        pass