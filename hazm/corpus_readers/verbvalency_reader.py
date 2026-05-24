"""This module includes classes and functions for reading the Persian Verb Valency Lexicon.

The Persian Verb Valency Lexicon is a collection containing valency information for
more than 4,500 Persian verbs. This lexicon specifies obligatory and optional
complements for various types of verbs: simple, compound, prefixed, and phrasal
verbs. The high frequency of compound verbs in Persian doubles the need for a
verb valency lexicon, as identifying compound verbs is more difficult than
identifying simple ones for both humans and machines. Providing a list of verbs
(including compound verbs) along with their valency structures is a significant
help for NLP tasks. Furthermore, based on Dependency Theory, the fundamental
structure of a sentence can be derived from the verb's valency, which adds to the
importance of knowing these structures in linguistic texts.
"""


from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


class Verb(NamedTuple):
    """A named tuple representing a Persian verb and its valency properties.

    Attributes:
        past_light_verb: The past light verb.
        present_light_verb: The present light verb.
        prefix: The verb prefix.
        nonverbal_element: The non-verbal element of a compound verb.
        preposition: The associated preposition.
        valency: The valency structure of the verb.
    """
    past_light_verb: str
    present_light_verb: str
    prefix: str
    nonverbal_element: str
    preposition: str
    valency: str


class VerbValencyReader:
    """This class includes functions for reading the Persian Verb Valency Lexicon.

    Args:
        valency_file: Path to the lexicon file.
    """

    def __init__(
        self: "VerbValencyReader", valency_file: str = "valency.txt",
    ) -> None:
        """Initializes the VerbValencyReader.

        Args:
            valency_file: Path to the lexicon file. Defaults to "valency.txt".
        """
        self._valency_file = valency_file

    def verbs(self: "VerbValencyReader") -> Iterator[Verb]:
        """Iterates through the verbs in the lexicon.

        Yields:
            The next verb in the lexicon as a Verb object.
        """
        with Path.open(self._valency_file, encoding="utf-8") as valency_file:
            for line in valency_file:
                if "بن ماضی" in line:
                    continue

                line = line.strip().replace("-\t", "\t")
                parts = line.split("\t")
                if len(parts) == 6:
                    yield Verb(*parts)
