"""This module includes classes and functions for reading the Wikipedia corpus.

The [Wikipedia corpus](http://download.wikimedia.org/fawiki/latest/fawiki-latest-pages-articles.xml.bz2)
is a massive corpus containing all Persian Wikipedia articles, updated every two months.
For more information about this corpus, you can visit its
[main page](https://dumps.wikimedia.org/backup-index.html).
"""


import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path


class WikipediaReader:
    """This class includes functions for reading the Wikipedia corpus.

    Args:
        fawiki_dump: Path to the corpus dump file.
        n_jobs: Number of CPU cores for parallel processing.
    """

    def __init__(self: "WikipediaReader", fawiki_dump: str, n_jobs: int = 2) -> None:
        """Initializes the Wikipedia reader.

        Args:
            fawiki_dump: Path to the corpus dump file.
            n_jobs: Number of CPU cores for parallel processing.
        """
        self.fawiki_dump = fawiki_dump
        self.wiki_extractor = Path(__file__).parent / "wiki_extractor.py"
        self.n_jobs = n_jobs

    def docs(self: "WikipediaReader") -> Iterator[dict[str, str]]:
        """Yields articles from the corpus.

        Each article is a dictionary containing the following parameters:
        - id: The article identifier.
        - title: The title of the article.
        - text: The content of the article.
        - date: The web version date.
        - url: The page URL.

        Examples:
            >>> wikipedia = WikipediaReader('fawiki-latest-pages-articles.xml.bz2')
            >>> next(wikipedia.docs())['id']

        Yields:
            A dictionary containing the next article's data.
        """
        proc = subprocess.Popen(
            [
                "python",
                self.wiki_extractor,
                "--no-templates",
                "--processes",
                str(self.n_jobs),
                "--output",
                "-",
                self.fawiki_dump,
            ],
            stdout=subprocess.PIPE,
        )
        doc_pattern = re.compile(r'<doc id="(\d+)" url="([^\"]+)" title="([^\"]+)">')

        doc = []
        for line in iter(proc.stdout.readline, b""):
            line = line.strip().decode("utf8")
            if line:
                doc.append(line)

            if line == "</doc>":
                del doc[1]
                id, url, title = doc_pattern.match(doc[0]).groups()  # noqa: A001
                html = "\n".join(doc[1:-1])

                yield {"id": id, "url": url, "title": title, "html": html, "text": html}
                doc = []

    def texts(self: "WikipediaReader") -> Iterator[str]:
        """Yields only the text of the articles.

        This function is provided for convenience. It is equivalent to using the
        [docs()][hazm.corpus_readers.wikipedia_reader.WikipediaReader.docs] method
        and retrieving the `text` property.

        Examples:
            >>> wikipedia = WikipediaReader('fawiki-latest-pages-articles.xml.bz2')
            >>> next(wikipedia.texts())[:30]

        Yields:
            The text of the next article.
        """
        for doc in self.docs():
            yield doc["text"]
