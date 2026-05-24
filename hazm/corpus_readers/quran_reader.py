"""This module includes classes and functions for reading the Quranic Arabic corpus.

The [Quranic Arabic](https://corpus.quran.com/) corpus contains syntactic rules
and morphological information for every word in the Holy Quran.
"""
from collections.abc import Iterator
from pathlib import Path

from ..utils import maketrans

buckwalter_transliteration = maketrans(
    "'>&<}AbptvjHxd*rzs$SDTZEg_fqklmnhwYyFNKaui~o^#`{:@\"[;,.!-+%]",
    "\u0621\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064a\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0670\u0671\u06dc\u06df\u06e0\u06e2\u06e3\u06e5\u06e6\u06e8\u06ea\u06eb\u06ec\u06ed",
)


class QuranReader:
    """This class includes functions for reading the Quranic Arabic corpus.

    Args:
        quran_file: Path to the corpus file.
    """

    def __init__(self: "QuranReader", quran_file: str) -> None:
        """Initializes the QuranReader with the given file path.

        Args:
            quran_file: Path to the corpus file.
        """
        self._quran_file = quran_file

    def parts(self: "QuranReader") -> Iterator[dict[str, str]]:
        """Yields the parts of the Quranic text along with their syntactic information.

        A part is not necessarily a word; for example, the word "Ar-Rahman" is
        composed of two parts: "Al" and "Rahman".

        Examples:
            >>> parts = QuranReader(quran_file='quranic_corpus_morphology.txt').parts()
            >>> print(next(parts))
            {'loc': (1, 1, 1, 1), 'text': 'بِ', 'tag': 'P'}
            >>> print(next(parts))
            {'loc': (1, 1, 1, 2), 'text': 'سْمِ', 'tag': 'N', 'lem': 'ٱسْم', 'root': 'سمو'}
            >>> print(next(parts))
            {'loc': (1, 1, 2, 1), 'text': 'ٱللَّهِ', 'tag': 'PN', 'lem': 'ٱللَّه', 'root': 'اله'}

        Yields:
            The next part of the Quranic text.
        """
        with Path(self._quran_file).open(encoding="utf8") as file:
            for line in file:
                if not line.startswith("("):
                    continue
                parts = line.strip().split("\t")

                part = {
                    "loc": eval(parts[0].replace(":", ",")),
                    "text": parts[1].translate(buckwalter_transliteration),
                    "tag": parts[2],
                }

                features = parts[3].split("|")
                for feature in features:
                    if feature.startswith("LEM:"):
                        part["lem"] = feature[4:].translate(buckwalter_transliteration)
                    elif feature.startswith("ROOT:"):
                        part["root"] = feature[5:].translate(buckwalter_transliteration)
                yield part

    def words(
        self: "QuranReader",
    ) -> Iterator[tuple[str, str, str, str, str, list[dict[str, str]]]]:
        """Yields morphological information for the words of the Quran.

        Examples:
            >>> words = QuranReader(quran_file='quranic_corpus_morphology.txt').words()
            >>> print(next(words))
            ('1.1.1', 'بِسْمِ', 'ٱسْم', 'سمو', 'P-N', [{'text': 'بِ', 'tag': 'P'}, {'text': 'سْمِ', 'tag': 'N', 'lem': 'ٱسْم', 'root': 'سمو'}])

        Yields:
            Morphological information of the next word in the Quran.
        """

        def word_item(location: tuple[int], parts: list[dict]) -> str:
            """Formats word-level information from its constituent parts.

            Args:
                location: A tuple representing the location (chapter, verse, word).
                parts: A list of dictionaries, where each dictionary represents a part of the word.

            Returns:
                A tuple containing:
                    - Formatted location string (e.g., '1.1.1').
                    - Combined text of the word.
                    - Combined lemmas.
                    - Combined roots.
                    - Combined tags.
                    - List of parts dictionaries.
            """
            text = "".join([part["text"] for part in parts])
            tag = "-".join([part["tag"] for part in parts])
            lem = "-".join([part["lem"] for part in parts if "lem" in part])
            root = "-".join([part["root"] for part in parts if "root" in part])
            return ".".join(map(str, location)), text, lem, root, tag, parts

        last_location = (0, 0, 0, 0)
        items = []
        for part in self.parts():
            if last_location[:3] == part["loc"][:3]:
                items.append(part)
            else:
                if items:
                    yield word_item(last_location[:3], items)
                items = [part]
            last_location = part["loc"]
            del part["loc"]
        yield word_item(last_location[:3], items)
