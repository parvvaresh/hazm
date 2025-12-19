"""This module includes classes and functions for reading the [PerUDT](https://github.com/phsfr/UD_Persian-PerDT) corpus.

PerUDT contains a significant number of labeled sentences with syntactic and
morphological information.
"""
import sys
from collections.abc import Iterator
from pathlib import Path

from .dadegan_reader import DadeganReader


def conllu2conll(conllu_path: str) -> str :
    """Converts a CoNLL-U file to the old CoNLL format.

    Args:
        conllu_path: Path to the CoNLL-U file.

    Returns:
        The content of the file converted to CoNLL format as a string.
    """
    delex = False
    if len(sys.argv) > 3 and sys.argv[3] == "delex":
        delex = True

    lines = []

    with Path(conllu_path).open(encoding="utf8") as reader1:
        line1 = reader1.readline()

        while line1:
            if len(line1.strip()) == 0:
                lines.append(line1)
            else:
                spl = line1.strip().split("\t")
                if len(spl) > 2 and "." not in spl[0] and spl[0].isdigit():
                    if ":" in spl[7]:
                        spl[7] = spl[7][:spl[7].rfind(":")]
                    if spl[6] == "_" or spl[6] == "-":
                        spl[6] = "-1"
                    if delex:
                        spl[1] = "_"
                        spl[2] = "_"
                    lines.append("\t".join(spl) + "\n")

            line1 = reader1.readline()

    return "".join(lines)

class UniversalDadeganReader(DadeganReader):
    """This class includes functions for reading the PerUDT corpus.

    Args:
        conllu_file: Path to the CoNLL-U corpus file.
    """
    def __init__(self: DadeganReader, conllu_file: str) -> None:
        """Initializes the UniversalDadeganReader.

        Args:
            conllu_file: Path to the CoNLL-U corpus file.
        """
        self._conll_file = conllu_file
        self._pos_map = lambda tags, _: ",".join(tags)
        self._top_relation_label = "root"

    def _sentences(self: DadeganReader) -> Iterator[str]:
        """Yields sentences of the corpus in raw text format.

        Yields:
            The next sentence in the corpus.
        """
        text = conllu2conll(self._conll_file)

        # refine text
        text = text.replace("‌‌", "‌").replace("\t‌", "\t").replace("‌\t", "\t").replace("\t ", "\t").replace(" \t", "\t").replace(
            "\r", "").replace("\u2029", "‌")

        for item in text.replace(" ", "_").split("\n\n"):
            if item.strip():
                yield item
