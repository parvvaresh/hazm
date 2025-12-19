"""This module includes classes and functions for reading the Mizan corpus.

The [Mizan corpus](https://github.com/omidkashefi/Mizan/) contains more than 1 million
English sentences (mostly in the field of classical literature) and their Persian
translations, prepared by the Secretariat of the Supreme Council of Information
and Communication Technology.
"""
from collections.abc import Iterator
from pathlib import Path


class MizanReader:
    """A reader for the Mizan corpus.

    Args:
        corpus_folder: Path to the folder containing the Mizan corpus files.
    """
    def __init__(self: "MizanReader", corpus_folder: str) -> None:
        """Initializes the Mizan reader.

        Args:
            corpus_folder: Path to the folder containing the Mizan corpus files.
        """
        self._corpus_folder = Path(corpus_folder)
        self._en_file_path = self._corpus_folder / "mizan_en.txt"
        self._fa_file_path = self._corpus_folder / "mizan_fa.txt"


    def english_sentences(self: "MizanReader") -> Iterator[str]:
        """Yields English sentences one by one.

        Examples:
            >>> mizan = MizanReader("mizan")
            >>> next(mizan.english_sentences())
            'The story which follows was first written out in Paris during the Peace Conference'

        Yields:
            The next English sentence.
        """
        with Path(self._en_file_path).open("r", encoding="utf-8") as file:
            for line in file:
                    yield line.strip()


    def persian_sentences(self: "MizanReader") -> Iterator[str]:
        """Yields Persian sentences one by one.

        Examples:
            >>> mizan = MizanReader("mizan")
            >>> next(mizan.persian_sentences())
            'داستانی که از نظر شما می‌گذرد، ابتدا ضمن کنفرانس صلح پاریس از روی یادداشت‌هائی که به طور روزانه در حال خدمت در صف برداشته شده بودند'

        Yields:
            The next Persian sentence.
        """
        with Path(self._fa_file_path).open("r", encoding="utf-8") as file:
            for line in file:
                    yield line.strip()

    def english_persian_sentences(self: "MizanReader") -> Iterator[tuple[str, str]]:
        r"""Yields pairs of English and Persian sentences side by side.

        Examples:
            >>> mizan = MizanReader("mizan")
            >>> next(mizan.english_persian_sentences())
            ('The story which follows was first written out in Paris during the Peace Conference', 'داستانی که از نظر شما می\\u200cگذرد، ابتدا ضمن کنفرانس صلح پاریس از روی یادداشت\\u200cهائی که به طور روزانه در حال خدمت در صف برداشته شده بودند')

        Yields:
            A tuple of (English sentence, Persian sentence).
        """
        yield from zip(self.english_sentences(), self.persian_sentences(), strict=False)
