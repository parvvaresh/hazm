"""This module includes classes and functions for reading the FAspell corpus.

The [FAspell](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-1547)
corpus contains 5,063 Persian spelling errors. This corpus also includes 801
misidentifications by OCR systems.
"""
from collections.abc import Iterator
from pathlib import Path


class FaSpellReader:
    """This class includes functions for reading the FAspell corpus.

    Args:
        corpus_folder: Path to the folder containing the corpus files.
    """

    def __init__(self: "FaSpellReader", corpus_folder: str) -> None:
        """Initializes the FAspell reader.

        Args:
            corpus_folder: Path to the folder containing the corpus files.
        """
        self._corpus_folder = Path(corpus_folder)
        self._main_file_path = self._corpus_folder / "faspell_main.txt"
        self._ocr_file_path = self._corpus_folder / "faspell_ocr.txt"

    def main_entries(self: "FaSpellReader") -> Iterator[tuple[str, str, int]]:
        """Yields misspelled words, their correct forms, and error categories.

        Each entry is returned as a tuple: (misspelled_form, correct_form, error_category).

        Examples:
            >>> faspell = FaSpellReader(corpus_folder='faspell')
            >>> next(faspell.main_entries())
            ("آاهي", "آگاهی", 1)

        Yields:
            The next entry in the main corpus.
        """
        with Path(self._main_file_path).open("r", encoding="utf-8") as file:
            next(file) # skip the first line (header line)
            for line in file:
                parts = line.strip().split("\t")
                misspelt, corrected, error_category = parts
                yield (misspelt, corrected, int(error_category))

    def ocr_entries(self: "FaSpellReader") -> Iterator[tuple[str, str]]:
        """Yields OCR-misidentified words and their correct equivalents.

        Each entry is returned as a tuple: (misidentified_form, correct_form).

        Examples:
            >>> faspell = FaSpellReader(corpus_folder='faspell')
            >>> next(faspell.ocr_entries())
            ("آمدیم", "آ!دبم")

        Yields:
            The next OCR entry in the corpus.
        """
        with Path(self._ocr_file_path).open("r", encoding="utf-8") as file:
            next(file) # skip the first line (header line)
            for line in file:
                parts = line.strip().split("\t")
                misspelt, corrected = parts
                yield (misspelt, corrected)
