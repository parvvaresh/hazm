"""This module includes classes and functions for reading the MirasText corpus.

[MirasText](https://github.com/miras-tech/MirasText) contains 2,835,414 news
items from 250 Persian news agencies.
"""

from collections.abc import Iterator
from pathlib import Path


class MirasTextReader:
    """This class includes functions for reading the MirasText corpus.

    Args:
        filename: Path to the corpus file.
    """

    def __init__(self: "MirasTextReader", filename: str) -> None:
        """Initializes the MirasText reader.

        Args:
            filename: Path to the corpus file.
        """
        self._filename = filename

    def docs(self: "MirasTextReader") -> Iterator[dict[str, str]]:
        """Yields news documents.

        Yields:
            The next news document.
        """
        with Path(self._filename).open(encoding="utf-8") as file:
            for line in file:
                parts = line.split("***")
                # todo: extract link, tags, ...
                yield {"text": parts[0].strip()}

    def texts(self: "MirasTextReader") -> Iterator[str]:
        """Yields only the text of the news articles.

        This method is provided for convenience; the same result can be achieved
        by using [docs()][hazm.corpus_readers.mirastext_reader.MirasTextReader.docs]
        and accessing the `text` key.

        Examples:
            >>> mirastext = MirasTextReader(filename='mirastext.txt')
            >>> next(mirastext.texts())[:42]  # first 42 characters of first text
            'ایرانی‌ها چقدر از اینترنت استفاده می‌کنند؟'

        Yields:
            The text of the next news article.
        """
        for doc in self.docs():
            yield doc["text"]
