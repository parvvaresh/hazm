"""This module includes classes and functions for reading the SentiPers corpus.

SentiPers contains a collection of Persian texts with semantic labels.
"""


import itertools
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from xml.dom import minidom


class SentiPersReader:
    """This class includes functions for reading the SentiPers corpus.

    Args:
        root: Path to the folder containing the corpus files.
    """

    def __init__(self: "SentiPersReader", root: str) -> None:
        """Initializes the SentiPers reader.

        Args:
            root: Path to the folder containing the corpus files.
        """
        self._root = root

    def docs(self: "SentiPersReader") -> Iterator[dict[str, Any]]:
        """Yields documents from the SentiPers corpus.

        Each document is returned as a dictionary containing these fields:
        - Title
        - Type
        - comments: A list of comment dictionaries.

        Each dictionary in the `comments` list includes:
        - id
        - type
        - author
        - value
        - sentences: A list of sentence dictionaries (text, id, value).

        Yields:
            The next document in the corpus.
        """

        def element_sentences(element: Any) -> Iterator[dict[str, Any]]:
            """Extracts sentences from an XML element.

            Args:
                element: The XML element to extract sentences from.

            Yields:
                A dictionary for each sentence containing 'text', 'id', and 'value'.
            """
            for sentence in element.getElementsByTagName("Sentence"):
                yield {
                    "text": sentence.childNodes[0].data,
                    "id": sentence.getAttribute("ID"),
                    "value": (
                        int(sentence.getAttribute("Value"))
                        if comment.getAttribute("Value")
                        else None
                    ),
                }

        for root, _dirs, files in os.walk(self._root):
            for filename in sorted(files):
                try:
                    elements = minidom.parse(os.path.join(root, filename)) # noqa: PTH118

                    product = elements.getElementsByTagName("Product")[0]
                    doc = {
                        "Title": product.getAttribute("Title"),
                        "Type": product.getAttribute("Type"),
                        "comments": [],
                    }

                    for child in product.childNodes:
                        if child.nodeName in {
                            "Voters",
                            "Performance",
                            "Capability",
                            "Production_Quality",
                            "Ergonomics",
                            "Purchase_Value",
                        }:
                            value = child.getAttribute("Value")
                            doc[child.nodeName] = (
                                float(value) if "." in value else int(value)
                            )

                    for comment in itertools.chain(
                        elements.getElementsByTagName("Opinion"),
                        elements.getElementsByTagName("Criticism"),
                    ):
                        doc["comments"].append(
                            {
                                "id": comment.getAttribute("ID"),
                                "type": comment.nodeName,
                                "author": comment.getAttribute("Holder").strip(),
                                "value": (
                                    int(comment.getAttribute("Value"))
                                    if comment.getAttribute("Value")
                                    else None
                                ),
                                "sentences": list(element_sentences(comment)),
                            },
                        )

                    # todo: Accessories, Features, Review, Advantages, Tags, Keywords, Index

                    yield doc

                except Exception as e:
                    print("error in reading", filename, e, file=sys.stderr)

    def comments(self: "SentiPersReader") -> Iterator[list[list[str]]]:
        """Yields comments belonging to each document.

        Examples:
            >>> sentipers = SentiPersReader(root='sentipers')
            >>> next(sentipers.comments())[0][1]
            'بيشتر مناسب است براي کساني که به دنبال تنوع هستند و در همه چيز نو گرايي دارند .'

        Yields:
            A list of comments for the next document, where each comment is a list of its sentences.
        """
        for doc in self.docs():
            yield [
                [sentence["text"] for sentence in text]
                for text in [comment["sentences"] for comment in doc["comments"]]
            ]
