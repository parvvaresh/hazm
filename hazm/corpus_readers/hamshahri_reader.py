"""This module includes classes and functions for reading the Hamshahri corpus.

The [Hamshahri Corpus](https://www.peykaregan.ir/dataset/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D9%87%D9%85%D8%B4%D9%87%D8%B1%DB%8C)
contains 318,000 news items from the Hamshahri newspaper from 1996 to 2007 (1375 to 1386 AP).
This data was prepared by crawling the Hamshahri website and undergoing several stages
of preprocessing and labeling. All news items have a CAT label, and their thematic
classification is specified. This corpus was prepared by the Database Research
Group of the University of Tehran with the support of the Iran Telecommunication
Research Center (ITRC).
"""


import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from xml.dom import minidom


class HamshahriReader:
    """This class includes functions for reading the Hamshahri corpus.

    Args:
        root: Path to the folder containing the Hamshahri corpus files.
    """

    def __init__(self: "HamshahriReader", root: str) -> None:
        """Initializes the Hamshahri reader.

        Args:
            root: Path to the folder containing the Hamshahri corpus files.
        """
        self._root = root
        self._invalids = {
            "hamshahri.dtd",
            "HAM2-960622.xml",
            "HAM2-960630.xml",
            "HAM2-960701.xml",
            "HAM2-960709.xml",
            "HAM2-960710.xml",
            "HAM2-960711.xml",
            "HAM2-960817.xml",
            "HAM2-960818.xml",
            "HAM2-960819.xml",
            "HAM2-960820.xml",
            "HAM2-961019.xml",
            "HAM2-961112.xml",
            "HAM2-961113.xml",
            "HAM2-961114.xml",
            "HAM2-970414.xml",
            "HAM2-970415.xml",
            "HAM2-970612.xml",
            "HAM2-970614.xml",
            "HAM2-970710.xml",
            "HAM2-970712.xml",
            "HAM2-970713.xml",
            "HAM2-970717.xml",
            "HAM2-970719.xml",
            "HAM2-980317.xml",
            "HAM2-040820.xml",
            "HAM2-040824.xml",
            "HAM2-040825.xml",
            "HAM2-040901.xml",
            "HAM2-040917.xml",
            "HAM2-040918.xml",
            "HAM2-040920.xml",
            "HAM2-041025.xml",
            "HAM2-041026.xml",
            "HAM2-041027.xml",
            "HAM2-041230.xml",
            "HAM2-041231.xml",
            "HAM2-050101.xml",
            "HAM2-050102.xml",
            "HAM2-050223.xml",
            "HAM2-050224.xml",
            "HAM2-050406.xml",
            "HAM2-050407.xml",
            "HAM2-050416.xml",
        }
        self._paragraph_pattern = re.compile(r"(\n.{0,50})(?=\n)")

    def docs(self: "HamshahriReader") -> Iterator[dict[str, str]]:
        """Yields news documents from the corpus.

        Each news item is a dictionary containing the following keys:
        - `id`: Unique identifier.
        - `title`: News title.
        - `text`: News content.
        - `issue`: Issue number.
        - `categories_{lang}`: Thematic categories (e.g., `categories_fa`).
        - `date`: News date (Persian).

        Examples:
            >>> hamshahri = HamshahriReader(root='hamshahri')
            >>> next(hamshahri.docs())['id']
            'HAM2-750403-001'

        Yields:
            The next news document dictionary.
        """
        for root, _dirs, files in os.walk(self._root):
            for name in sorted(files):
                if name in self._invalids:
                    continue

                try:
                    elements = minidom.parse(os.path.join(root, name)) # noqa: PTH118
                    for element in elements.getElementsByTagName("DOC"):
                        doc = {
                            "id": (
                                element.getElementsByTagName("DOCID")[0]
                                .childNodes[0]
                                .data
                            ),
                            "issue": (
                                element.getElementsByTagName("ISSUE")[0]
                                .childNodes[0]
                                .data
                            ),
                        }

                        for cat in element.getElementsByTagName("CAT"):
                            doc["categories_" + cat.attributes["xml:lang"].value] = (
                                cat.childNodes[0].data.split(".")
                            )

                        for date in element.getElementsByTagName("DATE"):
                            if date.attributes["calender"].value == "Persian":
                                doc["date"] = date.childNodes[0].data

                        elm = element.getElementsByTagName("TITLE")[0]
                        doc["title"] = (
                            elm.childNodes[1].data if len(elm.childNodes) > 1 else ""
                        )

                        doc["text"] = ""
                        for item in element.getElementsByTagName("TEXT")[0].childNodes:
                            if item.nodeType == 4:  # CDATA
                                doc["text"] += item.data

                        # refine text
                        doc["text"] = self._paragraph_pattern.sub(
                            r"\1\n",
                            doc["text"],
                        ).replace("\no ", "\n")

                        yield doc

                except Exception as e:
                    print("error in reading", name, e, file=sys.stderr)

    def texts(self: "HamshahriReader") -> Iterator[str]:
        """Yields only the text content of the news items.

        This function is provided for convenience. The same result can be
        achieved by using the
        [docs()][hazm.corpus_readers.hamshahri_reader.HamshahriReader.docs]
        method and accessing the `text` field.

        Yields:
            The text content of the next news item.
        """
        for doc in self.docs():
            yield doc["text"]
