"""This module includes classes and functions for reading the Peykare corpus.

[Peykare](https://www.peykaregan.ir/dataset/%D9%BE%DB%8C%DA%A9%D8%B1%D9%87-%D9%85%D8%AA%D9%86%DB%8C-%D8%B2%D8%A8%D8%A7%D9%86-%D9%81%D8%A7%D8%B1%D8%B3%DB%8C)
is a collection of formal written and spoken Persian texts collected from real
sources such as newspapers, websites, and pre-typed documents, which have been
corrected and tagged. The volume of this data is approximately 100 million words,
gathered from various sources with a high degree of diversity. 10 million words
of this corpus have been manually tagged by linguistics students using 882
syntactic-semantic tags, and each file is classified by its subject and source.
This corpus, prepared by the Intelligent Signal Processing Research Center,
is suitable for use in training language models and other natural language
processing projects.
"""


import codecs
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..normalizer import Normalizer
from ..word_tokenizer import WordTokenizer


def coarse_pos_u(tags: list[str], word: str) -> list[str]:
    """Converts fine-grained tags to coarse-grained universal POS tags.

    Examples:
        >>> coarse_pos_u(['N','COM','SING'], 'الجزیره')
        'NOUN'

    Args:
        tags: List of fine-grained tags.
        word: The word to be converted to a universal tag.

    Returns:
        List of coarse-grained universal POS tags.
    """
    map_pos_to_upos = {
        "N": "NOUN",
        "V": "VERB",
        "AJ": "ADJ",
        "ADV": "ADV",
        "PRO": "PRON",
        "DET": "DET",
        "P": "ADP",
        "POSTP": "ADP",
        "NUM": "NUM",
        "CONJ": "CCONJ",
        "PUNC": "PUNCT",
        "CL": "NOUN",
        "INT": "INTJ",
        "RES": "NOUN",
    }
    sconj_list = {
        "که",
        "تا",
        "گرچه",
        "اگرچه",
        "چرا",
        "زیرا",
        "اگر",
        "چون",
        "چراکه",
        "هرچند",
        "وگرنه",
        "چنانچه",
        "والا",
        "هرچه",
        "ولو",
        "مگر",
        "پس",
        "چو",
        "چه",
        "بنابراین",
        "وقتی",
        "والّا",
        "انگاری",
        "هرچندكه",
        "درنتيجه",
        "اگه",
        "ازآنجاكه",
        "گر",
        "وگر",
        "وقتيكه",
        "تااينكه",
        "زمانيكه",
    }
    num_adj_list = {
        "نخست",
        "دوم",
        "اول",
        "پنجم",
        "آخر",
        "يازدهم",
        "نهم",
        "چهارم",
        "ششم",
        "پانزدهم",
        "دوازدهم",
        "هشتم",
        "صدم",
        "هفتم",
        "هفدهم",
        "آخرين",
        "سيزدهم",
        "يكم",
        "بيستم",
        "ويكم",
        "دوسوم",
        "شانزدهم",
        "هجدهم",
        "چهاردهم",
        "ششصدم",
        "ميليونيم",
        "وهفتم",
        "يازدهمين",
        "هيجدهمين",
        "واپسين",
        "چهلم",
        "هزارم",
        "وپنجم",
        "هيجدهم",
        "ميلياردم",
        "ميليونيوم",
        "تريليونيوم",
        "چهارپنجم",
        "دهگانه",
        "ميليونم",
        "اوّل",
        "سوّم",
    }
    try:
        old_pos = next(
            iter(set(tags) & {
                "N", "V", "AJ", "ADV", "PRO", "DET",
                "P", "POSTP", "NUM", "CONJ", "PUNC",
                "CL", "INT", "RES",
            }),
        )

        if old_pos == "CONJ" and word in sconj_list:
            return "SCONJ"
        if old_pos == "NUM" and word in num_adj_list:
            return "ADJ" + (",EZ" if "EZ" in tags else "")
        return map_pos_to_upos[old_pos] + (",EZ" if "EZ" in tags else "")
    except:
        return "NOUN"


def coarse_pos_e(tags: list[str], word:str) -> list[str]: # noqa: ARG001
    """Converts fine-grained tags to coarse-grained POS tags.

    Examples:
        >>> coarse_pos_e(['N','COM','SING'],'الجزیره')
        'N'

    Args:
        tags: List of fine-grained tags.
        word: The word associated with the tags.

    Returns:
        List of coarse-grained tags.
    """
    try:
        return next(
            iter(set(tags) & {
                "N", "V", "AJ", "ADV", "PRO", "DET",
                "P", "POSTP", "NUM", "CONJ", "PUNC",
                "CL", "INT", "RES",
            }),
        ) + (",EZ" if "EZ" in tags else "")

    except:
        return "N"


def join_verb_parts(sentence: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Joins multi-part verbs with an underscore character.

    Takes a sentence in the form of a list of `(token, tag)` tuples and joins
    tokens belonging to multi-part verbs using an underscore (_).

    Examples:
        >>> join_verb_parts([('اولین', 'AJ'), ('سیاره', 'Ne'), ('خارج', 'AJ'), ('از', 'P'), ('منظومه', 'Ne'), ('شمسی', 'AJ'), ('دیده', 'AJ'), ('شد', 'V'), ('.', 'PUNC')])
        [('اولین', 'AJ'), ('سیاره', 'Ne'), ('خارج', 'AJ'), ('از', 'P'), ('منظومه', 'Ne'), ('شمسی', 'AJ'), ('دیده_شد', 'V'), ('.', 'PUNC')]

    Args:
        sentence: Sentence as a list of `(token, tag)` tuples.

    Returns:
        A list of `(token, tag)` tuples where multi-part verbs are joined into a single token.
    """
    if not hasattr(join_verb_parts, "tokenizer"):
        join_verb_parts.tokenizer = WordTokenizer()
    before_verbs, after_verbs, verbe = (
        join_verb_parts.tokenizer.before_verbs,
        join_verb_parts.tokenizer.after_verbs,
        join_verb_parts.tokenizer.verbe,
    )

    result = [("", "")]
    for word in reversed(sentence):
        if word[0] in before_verbs or (
            result[-1][0] in after_verbs and word[0] in verbe
        ):
            result[-1] = (word[0] + "_" + result[-1][0], result[-1][1])
        else:
            result.append(word)
    return list(reversed(result[1:]))


class PeykareReader:
    """A reader for the Peykare corpus.

    Args:
        root: Path to the root folder containing corpus files.
        joined_verb_parts: If `True`, multi-part verbs will be returned as joined tokens.
        pos_map: A function to map fine-grained tags to coarse-grained ones.
        universal_pos: If `True`, uses the universal POS tagset.
    """

    def __init__(
        self: "PeykareReader",
        root: str,
        joined_verb_parts: bool = True,
        pos_map: str = coarse_pos_e,
        universal_pos: bool = False,
    ) -> None:
        """Initializes the PeykareReader.

        Args:
            root: Path to the folder containing the corpus files.
            joined_verb_parts: If `True`, multi-part verbs will be joined using an underscore.
            pos_map: A mapper for fine-grained to coarse-grained tags.
            universal_pos: If `True`, uses universal POS tags.
        """
        self._root = root
        if pos_map is None:
            self._pos_map = lambda tags: ",".join(tags)
        elif universal_pos:
            self._pos_map = coarse_pos_u
        else:
            self._pos_map = coarse_pos_e
        self._joined_verb_parts = joined_verb_parts
        self._normalizer = Normalizer(correct_spacing=False)

    def docs(self: "PeykareReader") -> Iterator[str]:
        """Returns documents as raw text.

        Yields:
            The raw text of the next document.
        """
        for root, _, files in os.walk(self._root):
            for name in sorted(files):
                with Path.open(
                    Path(root) / name,
                    encoding="windows-1256",
                ) as peykare_file:
                    text = peykare_file.read()
                    # Convert all EOL to CRLF
                    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
                    if text:
                        yield text

    def doc_to_sents(
        self: "PeykareReader", document: str,
    ) -> Iterator[list[tuple[str, str]]]:
        """Converts an input document into a list of sentences.

        Each sentence is a list of `(word, tag)` tuples.

        Args:
            document: The raw document text to be converted.

        Yields:
            The next sentence in the form of a list of `(word, tag)` tuples.
        """
        sentence = []
        for line in document.split("\r\n"):
            if not line:
                continue

            parts = line.split(" ")
            tags, word = parts[3], self._normalizer.normalize("‌".join(parts[4:]))

            if word and word != "#":
                sentence.append((word, tags))

            if parts[2] == "PUNC" and word in {"#", ".", "؟", "!"}:
                if len(sentence) > 1:
                    yield sentence
                sentence = []

    def sents(self: "PeykareReader") -> Iterator[list[tuple[str, str]]]:
        """Returns sentences of the corpus as a list of `(token, tag)` tuples.

        Examples:
            >>> peykare = PeykareReader(root='peykare')
            >>> next(peykare.sents())
            [('دیرزمانی', 'N'), ('از', 'P'), ('راه\u200cاندازی', 'N,EZ'), ('شبکه\u200cی', 'N,EZ'), ('خبر', 'N,EZ'), ('الجزیره', 'N'), ('نمی\u200cگذرد', 'V'), ('،', 'PUNC'), ('اما', 'CONJ'), ('این', 'DET'), ('شبکه\u200cی', 'N,EZ'), ('خبری', 'AJ,EZ'), ('عربی', 'N'), ('بسیار', 'ADV'), ('سریع', 'ADV'), ('توانسته', 'V'), ('در', 'P'), ('میان', 'N,EZ'), ('شبکه\u200cهای', 'N,EZ'), ('عظیم', 'AJ,EZ'), ('خبری', 'AJ'), ('و', 'CONJ'), ('بنگاه\u200cهای', 'N,EZ'), ('چندرسانه\u200cای', 'AJ,EZ'), ('دنیا', 'N'), ('خودی', 'N'), ('نشان', 'N'), ('دهد', 'V'), ('.', 'PUNC')]

        Yields:
            The next sentence in the form of a list of `(token, tag)` tuples.
        """

        # >>> peykare = PeykareReader(root='peykare', joined_verb_parts=False, pos_map=None)
        # >>> next(peykare.sents())
        def map_pos(item: str) -> tuple:
            return (item[0], self._pos_map(item[1].split(","), item[0]))

        for document in self.docs():
            for sentence in self.doc_to_sents(document):
                if self._joined_verb_parts:
                    sentence = join_verb_parts(sentence)

                yield list(map(map_pos, sentence))
