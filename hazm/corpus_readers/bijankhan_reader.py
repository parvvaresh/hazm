"""This module includes classes and functions for reading the Bijankhan corpus.

[Bijankhan Corpus](https://www.peykaregan.ir/dataset/%D9%BE%DB%8C%DA%A9%D8%B1%D9%87-%D8%A8%DB%8C%E2%80%8C%D8%AC%D9%86%E2%80%8C%D8%AE%D8%A7%D9%86)
is a collection of Persian texts containing more than 2.6 million words,
tagged with 550 types of POS tags. This corpus, prepared at the Intelligent
Signal Processing Research Center, also includes more than 4,300 thematic
tags such as political, historical, etc., for the texts.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from ..normalizer import Normalizer
from .peykare_reader import join_verb_parts

default_pos_map = {
    "ADJ": "ADJ",
    "ADJ_CMPR": "ADJ",
    "ADJ_INO": "ADJ",
    "ADJ_ORD": "ADJ",
    "ADJ_SIM": "ADJ",
    "ADJ_SUP": "ADJ",
    "ADV": "ADV",
    "ADV_EXM": "ADV",
    "ADV_I": "ADV",
    "ADV_NEGG": "ADV",
    "ADV_NI": "ADV",
    "ADV_TIME": "ADV",
    "AR": "AR",
    "CON": "CONJ",
    "DEFAULT": "DEFAULT",
    "DELM": "PUNC",
    "DET": "PREP",
    "IF": "IF",
    "INT": "INT",
    "MORP": "MORP",
    "MQUA": "MQUA",
    "MS": "MS",
    "N_PL": "N",
    "N_SING": "N",
    "NN": "NN",
    "NP": "NP",
    "OH": "OH",
    "OHH": "OHH",
    "P": "PREP",
    "PP": "PP",
    "PRO": "PR",
    "PS": "PS",
    "QUA": "QUA",
    "SPEC": "SPEC",
    "V_AUX": "V",
    "V_IMP": "V",
    "V_PA": "V",
    "V_PRE": "V",
    "V_PRS": "V",
    "V_SUB": "V",
}


class BijankhanReader:
    """This class includes methods for reading the Bijankhan corpus.

    Args:
        bijankhan_file: Path to the corpus file.
        joined_verb_parts: If `True`, joins multi-part verbs with an underscore.
        pos_map: A dictionary for converting fine-grained to coarse-grained POS tags.
    """

    def __init__(
        self: "BijankhanReader",
        bijankhan_file: str,
        joined_verb_parts: bool = True,
        pos_map: str | None = None,
    ) -> None:
        """Initializes the BijankhanReader with the corpus file and settings.

        Args:
            bijankhan_file: Path to the corpus file.
            joined_verb_parts: If `True`, joins multi-part verbs with an underscore.
            pos_map: A dictionary for converting fine-grained to coarse-grained POS tags.
        """
        if pos_map is None:
            pos_map = default_pos_map
        self._bijankhan_file = bijankhan_file
        self._joined_verb_parts = joined_verb_parts
        self._pos_map = pos_map
        self._normalizer = Normalizer(correct_spacing=False)

    def _sentences(self: "BijankhanReader") -> Iterator[list[tuple[str, str]]]:
        """Returns sentences of the corpus in raw text format.

        Yields:
            The next sentence as a list of (token, tag) tuples.
        """
        sentence = []
        with Path(self._bijankhan_file).open(encoding="utf-8") as f:
            length = 2
            for line in f:
                parts = re.split("  +", line.strip())
                if len(parts) == length:
                    word, tag = parts
                    if word not in ("#", "*"):
                        word = self._normalizer.normalize(word)
                        sentence.append((word if word else "_", tag))
                    if (
                        tag == "DELM"
                        and word in ("#", "*", ".", "؟", "!")
                        and sentence
                    ):
                        yield sentence
                        sentence = []

    def sents(self: "BijankhanReader") -> Iterator[list[tuple[str, str]]]:
        """Returns corpus sentences as a list of (token, tag) tuples.

        Examples:
            >>> bijankhan = BijankhanReader(bijankhan_file='bijankhan.txt')
            >>> next(bijankhan.sents())
            [('اولین', 'ADJ'), ('سیاره', 'N'), ('خارج', 'ADJ'), ('از', 'PREP'), ('منظومه', 'N'), ('شمسی', 'ADJ'), ('دیده_شد', 'V'), ('.', 'PUNC')]

        Yields:
            The next sentence as a list of (token, tag) tuples.
        """

        def map_poses(item: tuple[str, str]) -> tuple[str, str]:
            """Maps the POS tag of a single item based on pos_map.

            Args:
                item: A tuple of (word, tag).

            Returns:
                A tuple of (word, mapped_tag).
            """
            return (item[0], self._pos_map.get(item[1], item[1]))

        for sentence in self._sentences():
            if self._joined_verb_parts:
                sentence = join_verb_parts(sentence)
            yield list(map(map_poses, sentence))
