"""This module includes classes and functions for reading the TNews corpus."""

import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from xml.dom import minidom


class TNewsReader:
    """A class to read and iterate over the TNews corpus files.

    Args:
        root (str): Path to the directory containing the corpus files.
    """

    def __init__(self: "TNewsReader", root: str) -> None:
        """Initializes the TNewsReader with the root directory and a regex cleaner.

        Args:
            root (str): Path to the root folder of the corpus.
        """
        self._root = root
        self.cleaner = re.compile(r"<[^<>]+>")

    def docs(self: "TNewsReader") -> Iterator[dict[str, str]]:
        """Returns news articles as an iterator of dictionaries.

        Each news article is represented as a dictionary with the following keys:
        - id: Unique identifier.
        - title: The main title of the news.
        - pre-title: Text appearing before the title.
        - post-title: Text appearing after the title.
        - text: The full content of the article.
        - brief: A short summary of the article.
        - url: The source URL.
        - category: The news category or topic.
        - datetime: The publication date and time.

        Examples:
            >>> tnews = TNewsReader(root='tnews')
            >>> next(tnews.docs())['id']
            '14092303482300013653'

        Yields:
            dict[str, str]: A dictionary containing the metadata and content of the next news article.
        """

        def get_text(element: Any) -> str:
            """Extracts raw text from an XML element and removes HTML tags.

            Args:
                element: The XML element node.

            Returns:
                str: Cleaned text content.
            """
            raw_html = element.childNodes[0].data if element.childNodes else ""
            return re.sub(self.cleaner, "", raw_html)

        for root, _dirs, files in os.walk(self._root):
            for name in sorted(files):
                try:
                    path = Path(root) / name
                    content = path.read_text(encoding="utf8")

                    # Fix XML formatting issues by removing control characters and closing the root tag
                    content = (
                        re.sub(
                            r"[\x1B\b\x1A]",
                            "",
                            content,
                        ).replace(
                            "</TNews>",
                            "",
                        )
                        + "</TNews>"
                    )

                    elements = minidom.parseString(content)
                    for element in elements.getElementsByTagName("NEWS"):
                        doc = {}
                        doc["id"] = get_text(element.getElementsByTagName("NEWSID")[0])
                        doc["url"] = get_text(element.getElementsByTagName("URL")[0])
                        doc["datetime"] = get_text(
                            element.getElementsByTagName("UTCDATE")[0],
                        )
                        doc["category"] = get_text(
                            element.getElementsByTagName("CATEGORY")[0],
                        )
                        doc["pre-title"] = get_text(
                            element.getElementsByTagName("PRETITLE")[0],
                        )
                        doc["title"] = get_text(
                            element.getElementsByTagName("TITLE")[0],
                        )
                        doc["post-title"] = get_text(
                            element.getElementsByTagName("POSTTITLE")[0],
                        )
                        doc["brief"] = get_text(
                            element.getElementsByTagName("BRIEF")[0],
                        )
                        doc["text"] = get_text(
                            element.getElementsByTagName("DESCRIPTION")[0],
                        )
                        yield doc

                except Exception as e:
                    print("error in reading", name, e, file=sys.stderr)

    def texts(self: "TNewsReader") -> Iterator[str]:
        """Returns only the text content of the news articles.

        This is a convenience method. The same result can be achieved by iterating
        through [docs()][hazm.corpus_readers.tnews_reader.TNewsReader.docs] and
        accessing the 'text' key.

        Examples:
            >>> tnews = TNewsReader(root='tnews')
            >>> next(tnews.texts()).startswith('به گزارش "  شبکه اطلاع رسانی اینترنتی بوتیا  " به نقل از ارگ نیوز')
            True

        Yields:
            str: The text content of the next news article.
        """
        for doc in self.docs():
            yield doc["text"]
