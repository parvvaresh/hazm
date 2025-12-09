"""این ماژول شامل کلاس‌ها و توابعی برای شناساییِ وابستگی‌های دستوری متن است."""

import os
import tempfile
import logging
from typing import Iterator, List, Tuple, Any
from pathlib import Path

from nltk.parse import DependencyGraph
from nltk.parse.malt import MaltParser as NLTKMaltParser
from hazm.types import TaggedSentence, Sentence

logger = logging.getLogger(__name__)


class MaltParser(NLTKMaltParser):
    """این کلاس شامل توابعی برای شناسایی وابستگی‌های دستوری است."""

    def __init__(
        self,
        tagger: Any,
        lemmatizer: Any,
        working_dir: str = "universal_dependency_parser",
        model_file: str = "langModel.mco",
    ) -> None:
        self.tagger = tagger
        self.working_dir = working_dir
        self.mco = model_file
        self._malt_bin = os.path.join(working_dir, "malt.jar")
        self.lemmatize = (
            lemmatizer.lemmatize if lemmatizer else lambda w, t: "_"
        )

    def parse_sents(self, sentences: list[Sentence], verbose: bool = False) -> Iterator[DependencyGraph]:
        """گراف وابستگی را برمی‌گرداند."""
        tagged_sentences = self.tagger.tag_sents(sentences)
        return self.parse_tagged_sents(tagged_sentences, verbose)

    def parse_tagged_sents(
        self,
        sentences: list[TaggedSentence],
        verbose: bool = False,
    ) -> Iterator[DependencyGraph]:
        """گراف وابستگی‌ها را برای جملات ورودی برمی‌گرداند."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "malt_input.conll")
            output_path = os.path.join(temp_dir, "malt_output.conll")
            
            with open(input_path, "w", encoding="utf8") as input_file:
                for sentence in sentences:
                    for i, (word, tag) in enumerate(sentence, start=1):
                        word = word.strip() or "_"
                        lemma = self.lemmatize(word, tag) or "_"
                        input_file.write(
                            f"{i}\t{word.replace(' ', '_')}\t{lemma.replace(' ', '_')}\t{tag}\t{tag}\t_\t0\tROOT\t_\t_\n"
                        )
                    input_file.write("\n\n")

            cmd = [
                "java",
                "-jar",
                self._malt_bin,
                "-w",
                self.working_dir,
                "-c",
                self.mco,
                "-i",
                input_path,
                "-o",
                output_path,
                "-m",
                "parse",
            ]
            
            if self._execute(cmd, verbose) != 0:
                raise Exception(f"MaltParser parsing failed: {' '.join(cmd)}")

            with open(output_path, encoding="utf8") as output_file:
                content = output_file.read()
                for item in content.split("\n\n"):
                    if item.strip():
                        yield DependencyGraph(item, top_relation_label="root")