# hazm/chunker.py
"""این ماژول شامل کلاس‌ها و توابعی برای تجزیهٔ متن به عبارات اسمی، فعلی و حرف است."""

import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import List
from typing import Tuple

import spacy
from nltk.chunk import RegexpParser
from nltk.chunk import conlltags2tree
from nltk.chunk import tree2conlltags
from nltk.chunk.util import ChunkScore
from nltk.tree import Tree
from spacy.tokens import Doc
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from tqdm import tqdm

from hazm.types import ChunkedSentence
from hazm.sequence_tagger import IOBTagger
from hazm.pos_tagger import POSTagger
from hazm.types import Sentence
from hazm.types import TaggedSentence
from hazm.types import Token

logger = logging.getLogger(__name__)


def tree2brackets(tree: Tree) -> str:
    """خروجی درختی را به یک ساختار کروشه‌ای تبدیل می‌کند."""
    s, tag = "", ""
    for item in tree2conlltags(tree):
        word, pos, chunk = item
        if chunk[0] in {"B", "O"} and tag:
            s += tag + "] "
            tag = ""

        if chunk[0] == "B":
            tag = chunk.split("-")[1]
            s += "["
        s += word + " "

    if tag:
        s += tag + "] "

    return s.strip()


class Chunker(IOBTagger):
    """این کلاس شامل توابعی برای تقطیع متن، آموزش و ارزیابی مدل است."""

    def __init__(
        self,
        model: str | Path | None = None,
        data_maker: Any = None,
    ) -> None:
        """Constructor."""
        final_data_maker = data_maker if data_maker is not None else self.data_maker
        self.posTagger = POSTagger()
        super().__init__(model, final_data_maker)

    def data_maker(self, tokens: list[TaggedSentence]) -> list[list[dict[str, Any]]]:
        """تبدیل توکن‌ها به ویژگی‌ها."""
        words = [[word for word, _ in token] for token in tokens]
        tags = [[tag for _, tag in token] for token in tokens]
        return [
            [
                self.features(words=word_tokens, pos_tags=tag_tokens, index=index)
                for index in range(len(word_tokens))
            ]
            for word_tokens, tag_tokens in zip(words, tags, strict=False)
        ]

    def features(
        self,
        words: list[str],
        pos_tags: list[str],
        index: int,
    ) -> dict[str, Any]:
        """ویژگی‌های کلمه را برمی‌گرداند."""
        word_features = self.posTagger.features(words, index)
        word_features.update(
            {
                "pos": pos_tags[index],
                "prev_pos": "" if index == 0 else pos_tags[index - 1],
                "next_pos": "" if index == len(pos_tags) - 1 else pos_tags[index + 1],
            },
        )
        return word_features

    def train(
        self,
        trees: list[Tree],
        c1: float = 0.4,
        c2: float = 0.04,
        max_iteration: int = 400,
        verbose: bool = True,
        file_name: str = "chunker_crf.model",
        report_duration: bool = True,
    ) -> None:
        """آموزش مدل."""
        tagged_list = [tree2conlltags(tree) for tree in trees]
        return super().train(
            tagged_list,
            c1,
            c2,
            max_iteration,
            verbose,
            file_name,
            report_duration,
        )

    def parse(self, sentence: TaggedSentence) -> Tree:
        """درخت تقطیع‌شدهٔ جمله را بر می‌گرداند."""
        tagged = super().tag(sentence)
        return conlltags2tree(tagged)

    def parse_sents(self, sentences: list[TaggedSentence]) -> Iterator[Tree]:
        """جملات ورودی را به‌شکل تقطیع‌شده برمی‌گرداند."""
        for conlltagged in super().tag_sents(sentences):
            yield conlltags2tree(conlltagged)

    def evaluate(self, trees: list[Tree]) -> float:
        """دقت مدل را برمی‌گرداند."""
        tagged_sents = [tree2conlltags(tree) for tree in trees]
        return super().evaluate(tagged_sents)


class RuleBasedChunker(RegexpParser):
    """کلاس RuleBasedChunker."""

    def __init__(self) -> None:
        grammar = r"""
            NP:
                <P>{<N>}<V>

            VP:
                <.*[^e]>{<N>?<V>}
                {<V>}

            ADVP:
                {<ADVe?><AJ>?}

            ADJP:
                <.*[^e]>{<AJe?>}

            NP:
                {<DETe?|Ne?|NUMe?|AJe|PRO|CL|RESe?><DETe?|Ne?|NUMe?|AJe?|PRO|CL|RESe?>*}
                <N>}{<.*e?>

            ADJP:
                {<AJe?>}

            POSTP:
                {<POSTP>}

            PP:
                {<Pe?>+}
        """
        super().__init__(grammar=grammar)


class SpacyChunker(Chunker):
    """A Chunker based on the Spacy library."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        using_gpu: bool = False,
        gpu_id: int = 0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.model_path = str(model_path) if model_path else None
        self.using_gpu = using_gpu
        self.gpu_id = gpu_id
        self.model = None
        self.gpu_availability = False
        self.peykare_dict: dict[str, list[str]] = {}

        if self.model_path:
            self._setup()

    def _setup(self) -> None:
        if self.using_gpu:
            self._setup_gpu()
        else:
            logger.info("Using CPU for SpacyChunker.")

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

    def _setup_dataset(
        self,
        sents: list[ChunkedSentence],
        saved_directory: str,
        dataset_type: str,
    ) -> None:
        assert dataset_type in ["train", "dev", "test"]
        db = DocBin()
        for sent in tqdm(sents):
            words = [word[0] for word in sent]
            tags = [word[2] for word in sent] # Chunk tags
            # Note: Spacy usually expects POS tags in tag_ and NER/Chunk in ents or specific attributes.
            # Here we map chunk tags to tag_ attribute as per original code logic for simplicity,
            # though standard way is using ents for chunks.
            doc = Doc(Vocab(strings=words), words=words)
            for d, tag in zip(doc, tags, strict=False):
                d.tag_ = tag
            db.add(doc)

        path = Path(saved_directory)
        if not path.exists():
            path.mkdir(parents=True)

        db.to_disk(f"{saved_directory}/{dataset_type}.spacy")

    def train(
        self,
        train_dataset: list[ChunkedSentence],
        test_dataset: list[ChunkedSentence],
        data_directory: str,
        base_config_file: str,
        train_config_path: str,
        output_dir: str,
        use_direct_config: bool = False,
    ) -> None:
        """Train the spaCy chunker model."""
        if not use_direct_config:
            logger.info("Setting up training configuration...")
            self.train_config_file = train_config_path
            subprocess.run(
                f"python -m spacy init fill-config {base_config_file} {train_config_path}",
                check=False,
                shell=True,
            )
        else:
            self.train_config_file = train_config_path

        if train_dataset:
            self._setup_dataset(
                sents=train_dataset,
                saved_directory=data_directory,
                dataset_type="train",
            )

        if test_dataset:
            self._setup_dataset(
                sents=test_dataset,
                saved_directory=data_directory,
                dataset_type="test",
            )

        train_data = f"{data_directory}/train.spacy"
        test_data = f"{data_directory}/test.spacy"

        cmd = f"python -m spacy train {self.train_config_file} --output ./{output_dir} --paths.train ./{train_data} --paths.dev ./{test_data}"
        if self.gpu_availability:
            cmd += f" --gpu-id {self.gpu_id}"

        subprocess.run(cmd, check=False, shell=True)
        self.model_path = f"{output_dir}/model-last"

        if test_dataset:
            tokens_list = [[w for w, _, _ in sent] for sent in test_dataset]
            self._update_dictionary(tokens_list)
            self.model = spacy.load(self.model_path)
            self.model.tokenizer = self._custom_tokenizer

    def evaluate(self, test_sents: list[ChunkedSentence]) -> ChunkScore:
        """Score the accuracy of the chunker."""
        golds = test_sents
        # Extract sentence tuples for parsing
        test_inp = [
            [(word, tag) for word, tag, _ in sent]
            for sent in golds
        ]

        parsed = self.parse_sents(test_inp)
        preds_tree = list(parsed)
        golds_tree = [conlltags2tree(sent) for sent in golds]

        chunkscore = ChunkScore()
        for pred, correct in zip(preds_tree, golds_tree, strict=False):
            chunkscore.score(correct, pred)

        print("Accuracy:", chunkscore.accuracy())
        print("Precision:", chunkscore.precision())
        print("Recall:", chunkscore.recall())
        print("F_Score:", chunkscore.f_measure())

        return chunkscore

    def parse(self, sentence: TaggedSentence) -> Tree:
        """Parse a single sentence."""
        tokens = [w for w, _ in sentence]
        if self.model is None:
             raise ValueError("Model not loaded.")

        self._update_dictionary([tokens])

        doc = self.model(" ".join(tokens))
        words = [w for w, _ in sentence]
        tags = [tag for _, tag in sentence]
        preds = [w.tag_ for w in doc] # Assuming model predicts chunks in tag_

        chunk = list(zip(words, tags, preds, strict=False))
        return conlltags2tree(chunk)

    def parse_sents(
        self,
        sentences: list[TaggedSentence],
        batch_size: int = 128,
    ) -> Iterator[Tree]:
        """Parse multiple sentences."""
        tokens_list = [[w for w, _ in sent] for sent in sentences]
        if self.model is None:
             raise ValueError("Model not loaded.")

        self._update_dictionary(tokens_list)

        docs = list(
            self.model.pipe(
                (" ".join(sent) for sent in tokens_list),
                batch_size=batch_size,
            ),
        )

        for i, doc in enumerate(docs):
            words = [w for w, _ in sentences[i]]
            tags = [tag for _, tag in sentences[i]]
            preds = [w.tag_ for w in doc]
            chunk = list(zip(words, tags, preds, strict=False))
            yield conlltags2tree(chunk)
