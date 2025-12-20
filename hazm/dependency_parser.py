"""This module includes classes and functions for identifying grammatical dependencies in text."""

import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import spacy
from nltk.parse import DependencyGraph
from nltk.parse.malt import MaltParser as NLTKMaltParser
from spacy.tokens import Doc

from hazm.types import Sentence
from hazm.types import TaggedSentence

logger = logging.getLogger(__name__)


class MaltParser(NLTKMaltParser):
    """This class includes functions for identifying grammatical dependencies."""

    def __init__(
        self,
        tagger: Any,
        lemmatizer: Any,
        working_dir: str = "universal_dependency_parser",
        model_file: str = "langModel.mco",
        repo_id: str | None = None,
        model_filename: str | None = None,
    ) -> None:
        """Constructor.

        Examples:
            >>> from hazm import POSTagger, Lemmatizer
            >>> tagger = POSTagger(repo_id="roshan-research/hazm-pos-tagger", model_filename="pos_tagger.model")
            >>> lemmatizer = Lemmatizer()
            >>> # Loading from Hugging Face Hub
            >>> parser = MaltParser(tagger=tagger, lemmatizer=lemmatizer, repo_id="roshan-research/hazm-dependency-parser", model_filename="langModel.mco")
            >>> # Loading from a local model file
            >>> # parser = MaltParser(tagger=tagger, lemmatizer=lemmatizer, working_dir='universal_dependency_parser', model_file='langModel.mco')

        Args:
            tagger: The POS tagger instance.
            lemmatizer: The lemmatizer instance.
            working_dir: The directory containing the MaltParser jar and model (local mode).
            model_file: The name of the model file (e.g., 'langModel.mco').
            repo_id: Hugging Face repository ID (e.g., "roshan-research/hazm-dependency-parser").
            model_filename: Filename inside the repository (e.g., "langModel.mco").
        """
        self.tagger = tagger
        self.lemmatize = (
            lemmatizer.lemmatize if lemmatizer else lambda _w, _t: "_"
        )

        final_working_dir = working_dir
        final_model_file = model_file
        final_malt_bin = Path(working_dir) / "malt.jar"

        if repo_id and model_filename:
            try:
                from huggingface_hub import snapshot_download

                cache_dir = snapshot_download(repo_id=repo_id)

                final_working_dir = cache_dir
                final_model_file = model_filename
                final_malt_bin = Path(cache_dir) / "malt.jar"

            except ImportError as e:
                msg = "Please install `huggingface-hub` to use pretrained models from Hub."
                raise ImportError(msg) from e
            except Exception as e:
                msg = f"Failed to download model from {repo_id}: {e}"
                raise ValueError(msg) from e

        self.working_dir = final_working_dir
        self.mco = final_model_file
        self._malt_bin = final_malt_bin


    def parse_sents(self, sentences: list[Sentence], verbose: bool = False) -> Iterator[DependencyGraph]:
        """Returns the dependency graph.

        Examples:
            >>> graphs = parser.parse_sents([['من', 'به', 'مدرسه', 'رفتم', '.']])
            >>> graph = next(graphs)
            >>> print(graph.tree())
            (رفتم من (به مدرسه) .)

        Args:
            sentences: A list of sentences to be parsed.
            verbose: If True, prints verbose output.

        Yields:
            A dependency graph for each sentence.
        """
        tagged_sentences = self.tagger.tag_sents(sentences)
        return self.parse_tagged_sents(tagged_sentences, verbose)

    def parse_tagged_sents(
        self,
        sentences: list[TaggedSentence],
        verbose: bool = False,
    ) -> Iterator[DependencyGraph]:
        """Returns dependency graphs for input sentences.

        Examples:
            >>> tagged_sentences = [[('من', 'PRON'), ('به', 'ADP'), ('مدرسه', 'NOUN'), ('رفتم', 'VERB'), ('.', 'PUNCT')]]
            >>> graphs = parser.parse_tagged_sents(tagged_sentences)
            >>> print(next(graphs).tree())
            (رفتم من (به مدرسه) .)

        Args:
            sentences: A list of tagged sentences.
            verbose: If True, prints verbose output.

        Yields:
            A dependency graph for each sentence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "malt_input.conll"
            output_path = Path(temp_dir) / "malt_output.conll"

            with Path(input_path).open("w", encoding="utf8") as input_file:
                for sentence in sentences:
                    for i, (word, tag) in enumerate(sentence, start=1):
                        word = word.strip() or "_"
                        lemma = self.lemmatize(word, tag) or "_"
                        input_file.write(
                            f"{i}\t{word.replace(' ', '_')}\t{lemma.replace(' ', '_')}\t{tag}\t{tag}\t_\t0\tROOT\t_\t_\n",
                        )
                    input_file.write("\n\n")

            cmd = [
                "java",
                "-jar",
                str(self._malt_bin),
                "-w",
                str(self.working_dir),
                "-c",
                str(self.mco),
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "-m",
                "parse",
            ]

            if self._execute(cmd, verbose) != 0:
                msg = f"MaltParser parsing failed: {' '.join(cmd)}"
                raise Exception(msg)

            with Path(output_path).open(encoding="utf8") as output_file:
                content = output_file.read()
                for item in content.split("\n\n"):
                    if item.strip():
                        yield DependencyGraph(item, top_relation_label="root")

class SpacyDependencyParser(MaltParser):
    """A Dependency Parser based on the Spacy library."""

    def __init__(
        self,
        tagger: Any,
        lemmatizer: Any,
        model_path: str | Path | None = None,
        using_gpu: bool = False,
        gpu_id: int = 0,
        repo_id: str | None = None,
    ) -> None:
        """Initialize.

        Examples:
            >>> from hazm import POSTagger, Lemmatizer
            >>> tagger = POSTagger(repo_id="roshan-research/hazm-pos-tagger", model_filename="pos_tagger.model")
            >>> lemmatizer = Lemmatizer()
            >>> # Loading from Hugging Face Hub
            >>> parser = SpacyDependencyParser(tagger=tagger, lemmatizer=lemmatizer, repo_id="roshan-research/hazm-spacy-dependency-parser")
            >>> # Loading from a local model directory
            >>> # parser = SpacyDependencyParser(tagger=tagger, lemmatizer=lemmatizer, model_path='path/to/spacy_model')

        Args:
            tagger: The POS tagger instance.
            lemmatizer: The lemmatizer instance.
            model_path: Path to the local Spacy model.
            using_gpu: Whether to use GPU.
            gpu_id: The ID of the GPU to use.
            repo_id: Hugging Face repository ID.
        """
        self.tagger = tagger
        self.lemmatize = (
            lemmatizer.lemmatize if lemmatizer else lambda _w, _t: "_"
        )

        self.model_path = str(model_path) if model_path else None
        self.using_gpu = using_gpu
        self.gpu_id = gpu_id
        self.model = None
        self.gpu_availability = False

        if repo_id:
            try:
                from huggingface_hub import snapshot_download
                self.model_path = snapshot_download(repo_id=repo_id)
            except ImportError as e:
                msg = "Please install `huggingface-hub` to use pretrained models from Hub."
                raise ImportError(msg) from e
            except Exception as e:
                msg = f"Failed to download model from {repo_id}: {e}"
                raise ValueError(msg) from e

        self.peykare_dict: dict[str, list[str]] = {}

        if self.model_path:
            self._setup()

    def _setup(self) -> None:
        if self.using_gpu:
            self._setup_gpu()
        else:
            logger.info("Using CPU for SpacyDependencyParser.")

        if self.model_path and Path(self.model_path).exists():
             self.model = spacy.load(self.model_path)
             self.model.tokenizer = self._custom_tokenizer

    def _setup_gpu(self) -> None:
        logger.info("GPU Setup Process Started...")
        if spacy.prefer_gpu(self.gpu_id):
            logger.info("GPU is available and ready for use.")
            spacy.require_gpu(self.gpu_id)
            self.gpu_availability = True
        else:
            logger.warning("GPU is not available; spaCy will use CPU.")
            self.gpu_availability = False

    def _custom_tokenizer(self, text: str) -> Doc:
        if self.model and text in self.peykare_dict:
            return Doc(self.model.vocab, self.peykare_dict[text])
        msg = "No tokenization available for input."
        raise ValueError(msg)

    def _update_dictionary(self, sents: list[list[str]]) -> None:
        """Add sentences to dictionary."""
        for sent in sents:
            key = " ".join(sent)
            if key not in self.peykare_dict:
                self.peykare_dict[key] = sent

    def parse(self, sentence: list[str]) -> DependencyGraph:
        """Parse a single sentence.

        Examples:
            >>> graph = parser.parse(['من', 'به', 'مدرسه', 'رفتم', '.'])
            >>> print(graph.tree())
            (رفتم من (به مدرسه) .)

        Args:
            sentence: A list of words in the sentence.

        Returns:
            The parsed dependency graph.
        """
        return next(self.parse_sents([sentence]))

    def parse_sents(self, sentences: list[list[str]]) -> Iterator[DependencyGraph]:
        """Parse multiple sentences.

        Examples:
            >>> graphs = parser.parse_sents([['من', 'به', 'مدرسه', 'رفتم', '.']])
            >>> graph = next(graphs)
            >>> print(graph.tree())
            (رفتم من (به مدرسه) .)

        Args:
            sentences: A list of sentences, where each sentence is a list of words.

        Yields:
            The parsed dependency graph for each sentence.
        """
        if self.model is None:
             msg = "Model not loaded."
             raise ValueError(msg)

        cleaned_sentences = []
        for sent in sentences:
            if sent and isinstance(sent[0], tuple):
                cleaned_sentences.append([word for word, _ in sent])
            else:
                cleaned_sentences.append(sent)

        docs = []
        for tokens in cleaned_sentences:
            doc = Doc(self.model.vocab, words=tokens)
            for _name, proc in self.model.pipeline:
                doc = proc(doc)
            docs.append(doc)

        for doc in docs:
            conll_lines = []
            for token in doc:
                head_index = token.head.i + 1
                if token.i == token.head.i:
                    head_index = 0

                lemma = token.lemma_ if token.lemma_ else "_"
                pos = token.pos_ if token.pos_ else "_"
                tag = token.tag_ if token.tag_ else "_"
                dep = token.dep_ if token.dep_ else "_"

                line = f"{token.i + 1}\t{token.text}\t{lemma}\t{pos}\t{tag}\t_\t{head_index}\t{dep}\t_\t_"
                conll_lines.append(line)

            conll_str = "\n".join(conll_lines)
            yield DependencyGraph(conll_str, top_relation_label="root")
