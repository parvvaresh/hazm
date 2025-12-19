"""این ماژول شامل کلاس‌ها و توابعی برای شناساییِ وابستگی‌های دستوری متن است."""

import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import spacy
from spacy.tokens import Doc

from nltk.parse import DependencyGraph
from nltk.parse.malt import MaltParser as NLTKMaltParser

from hazm.types import Sentence
from hazm.types import TaggedSentence

logger = logging.getLogger(__name__)


class MaltParser(NLTKMaltParser):
    """این کلاس شامل توابعی برای شناسایی وابستگی‌های دستوری است."""

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
            lemmatizer.lemmatize if lemmatizer else lambda w, t: "_"
        )
        
        final_working_dir = working_dir
        final_model_file = model_file
        final_malt_bin = os.path.join(working_dir, "malt.jar")

        if repo_id and model_filename:
            try:
                from huggingface_hub import snapshot_download
                
                cache_dir = snapshot_download(repo_id=repo_id)
                
                final_working_dir = cache_dir
                final_model_file = model_filename
                final_malt_bin = os.path.join(cache_dir, "malt.jar")
                
            except ImportError:
                raise ImportError("Please install `huggingface-hub` to use pretrained models from Hub.")
            except Exception as e:
                raise ValueError(f"Failed to download model from {repo_id}: {e}")

        self.working_dir = final_working_dir
        self.mco = final_model_file
        self._malt_bin = final_malt_bin


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
                self._malt_bin,
                "-w",
                self.working_dir,
                "-c",
                self.mco,
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
        working_dir: str = ".",
        model_file: str = "",
        using_gpu: bool = False,
        gpu_id: int = 0,
        repo_id: str | None = None,
        model_filename: str | None = None,
    ) -> None:
        """Initialize."""
        self.tagger = tagger
        self.lemmatize = (
            lemmatizer.lemmatize if lemmatizer else lambda w, t: "_"
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
            except ImportError:
                raise ImportError("Please install `huggingface-hub` to use pretrained models from Hub.")
            except Exception as e:
                raise ValueError(f"Failed to download model from {repo_id}: {e}")

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
        raise ValueError("No tokenization available for input.")

    def _update_dictionary(self, sents: list[list[str]]) -> None:
        """Add sentences to dictionary."""
        for sent in sents:
            key = " ".join(sent)
            if key not in self.peykare_dict:
                self.peykare_dict[key] = sent

    def parse(self, sentence: list[str]) -> DependencyGraph:
        """Parse a single sentence."""
        return next(self.parse_sents([sentence]))

    def parse_sents(self, sentences: list[list[str]]) -> Iterator[DependencyGraph]:
        """Parse multiple sentences."""
        if self.model is None:
             raise ValueError("Model not loaded.")

        cleaned_sentences = []
        for sent in sentences:
            if sent and isinstance(sent[0], tuple):
                cleaned_sentences.append([word for word, _ in sent])
            else:
                cleaned_sentences.append(sent)

        docs = []
        for tokens in cleaned_sentences:
            doc = Doc(self.model.vocab, words=tokens)
            for name, proc in self.model.pipeline:
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

