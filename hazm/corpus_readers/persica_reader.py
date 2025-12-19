"""This module includes classes and functions for reading the Persica corpus.

The [Persica corpus](https://www.peykaregan.ir/dataset/%D9%BE%D8%B1%D8%B3%DB%8C%DA%A9%D8%A7-%D9%BE%DB%8C%DA%A9%D8%B1%D9%87-%D9%85%D8%AA%D9%88%D9%86-%D8%AE%D8%A8%D8%B1%DB%8C)
contains news articles extracted from the ISNA news agency in eleven categories:
sports, economics, culture, religion, history, politics, science, social,
education, judicial law, and health. This data has been preprocessed and is
ready for use in various natural language processing and data mining applications.
"""
from collections.abc import Iterator
from pathlib import Path


class PersicaReader:
    """This class includes functions for reading the Persica corpus.

    Args:
        csv_file: Path to the corpus file with a .csv extension.
    """

    def __init__(self: "PersicaReader", csv_file: str) -> None:
        """Initializes the Persica reader.

        Args:
            csv_file: Path to the corpus file with a .csv extension.
        """
        self._csv_file = csv_file

    def docs(self: "PersicaReader") -> Iterator[dict[str, str]]:
        """Yields news articles one by one.

        Each news article is a dictionary consisting of these parameters:
        - `id`: Unique identifier.
        - `title`: Title of the news.
        - `text`: Main body of the news.
        - `date`: Publication date.
        - `time`: Publication time.
        - `category`: Primary category.
        - `category2`: Secondary category.

        Examples:
            >>> persica = PersicaReader('persica.csv')
            >>> next(persica.docs())['id']
            843656

        Yields:
            A dictionary containing the next news article's metadata and content.
        """
        lines = []
        with Path(self._csv_file).open(encoding="utf-8-sig") as file:
            for current_line in file:
                current_line = current_line.strip()
                if current_line:
                    if current_line.endswith(","):
                        lines.append(current_line[:-1])
                    else:
                        lines.append(current_line)
                        yield {
                            "id": int(lines[0]),
                            "title": lines[1],
                            "text": lines[2],
                            "date": lines[3],
                            "time": lines[4],
                            "category": lines[5],
                            "category2": lines[6],
                        }
                        lines = []

    def texts(self: "PersicaReader") -> Iterator[str]:
        """Yields only the text content of the news articles.

        This function is provided for convenience; the same result can be
        achieved by using the [docs()][hazm.corpus_readers.persica_reader.PersicaReader.docs]
        method and accessing the `text` property.

        Examples:
            >>> persica = PersicaReader('persica.csv')
            >>> next(persica.texts()).startswith('وزير علوم در جمع استادان نمونه كشور گفت')
            True

        Yields:
            The text content of the next news article.
        """
        for doc in self.docs():
            yield doc["text"]
