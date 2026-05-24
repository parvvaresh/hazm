"""This module reads raw text corpora."""
from collections.abc import Callable
from typing import Any

from nltk.corpus import PlaintextCorpusReader
from nltk.corpus.reader import StreamBackedCorpusView
from nltk.corpus.reader import read_blankline_block

from ..sentence_tokenizer import SentenceTokenizer
from ..word_tokenizer import WordTokenizer


class PersianPlainTextReader(PlaintextCorpusReader):
    """A reader for Persian raw text corpora.

    This class extends NLTK's PlaintextCorpusReader to provide default
    tokenization suitable for the Persian language.

    Attributes:
        CorpusView: The class used to create a stream-backed view of the corpus.
    """

    CorpusView = StreamBackedCorpusView

    def __init__(
        self: "PersianPlainTextReader",
        root: str,
        fileids: list,
        word_tokenizer: Callable = WordTokenizer.tokenize,
        sent_tokenizer: Callable = SentenceTokenizer.tokenize,
        para_block_reader: Callable = read_blankline_block,
        encoding: str = "utf8",
    ) -> None:
        """Initializes the Persian text corpus reader.

        Args:
            root: The root directory of the corpus.
            fileids: A list of file identifiers or a glob pattern for the files.
            word_tokenizer: A function used to tokenize words.
                Defaults to WordTokenizer.tokenize.
            sent_tokenizer: A function used to tokenize sentences.
                Defaults to SentenceTokenizer.tokenize.
            para_block_reader: A function used to read paragraph blocks.
                Defaults to read_blankline_block.
            encoding: The character encoding of the corpus files.
                Defaults to "utf8".
        """
        super().__init__(
            root,
            fileids,
            word_tokenizer,
            sent_tokenizer,
            para_block_reader,
            encoding,
        )
