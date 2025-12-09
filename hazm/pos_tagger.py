"""این ماژول شامل کلاس‌ها و توابعی برای برچسب‌گذاری توکن‌هاست."""

import logging
from pathlib import Path
from typing import Any, List

import spacy
from nltk.tag import stanford
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from tqdm import tqdm

from hazm.sequence_tagger import SequenceTagger
from hazm.api import TaggerProtocol
from hazm.types import Sentence, TaggedSentence, Token

logger = logging.getLogger(__name__)

PUNCTUATION_LIST = [
    '"', "#", "(", ")", "*", ",", "-", ".", "/", ":", "[", "]",
    "«", "»", "،", ";", "?", "!",
]


class POSTagger(SequenceTagger, TaggerProtocol):
    """این کلاس‌ها شامل توابعی برای برچسب‌گذاری توکن‌هاست."""

    def __init__(
        self,
        model: str | Path | None = None,
        data_maker: Any = None,
        universal_tag: bool = False,
    ) -> None:
        """Constructor."""
        final_data_maker = data_maker if data_maker is not None else self.data_maker
        self.__is_universal = universal_tag
        super().__init__(model, final_data_maker)

    def __universal_converter(self, tagged_list: TaggedSentence) -> TaggedSentence:
        return [(word, tag.split(",")[0]) for word, tag in tagged_list]

    def __is_punc(self, word: str) -> bool:
        return word in PUNCTUATION_LIST

    def data_maker(self, tokens: list[Sentence]) -> list[list[dict[str, Any]]]:
        """تبدیل توکن‌ها به ویژگی‌ها."""
        return [
            [self.features(token, index) for index in range(len(token))]
            for token in tokens
        ]

    def features(self, sentence: Sentence, index: int) -> dict[str, Any]:
        """استخراج ویژگی‌های یک کلمه در جمله."""
        word = sentence[index]
        return {
            "word": word,
            "is_first": index == 0,
            "is_last": index == len(sentence) - 1,
            # *ix
            "prefix-1": word[0],
            "prefix-2": word[:2],
            "prefix-3": word[:3],
            "suffix-1": word[-1],
            "suffix-2": word[-2:],
            "suffix-3": word[-3:],
            # word
            "prev_word": "" if index == 0 else sentence[index - 1],
            "two_prev_word": "" if index <= 1 else sentence[index - 2],
            "next_word": "" if index == len(sentence) - 1 else sentence[index + 1],
            "two_next_word": (
                ""
                if index >= len(sentence) - 2
                else sentence[index + 2]
            ),
            # digit
            "is_numeric": word.isdigit(),
            "prev_is_numeric": "" if index == 0 else sentence[index - 1].isdigit(),
            "next_is_numeric": (
                "" if index == len(sentence) - 1 else sentence[index + 1].isdigit()
            ),
            # punc
            "is_punc": self.__is_punc(word),
            "prev_is_punc": "" if index == 0 else self.__is_punc(sentence[index - 1]),
            "next_is_punc": (
                ""
                if index == len(sentence) - 1
                else self.__is_punc(sentence[index + 1])
            ),
        }

    def tag(self, tokens: Sentence) -> TaggedSentence:
        """یک جمله را برچسب‌گذاری می‌کند."""
        tagged_token = super().tag(tokens)
        return (
            self.__universal_converter(tagged_token)
            if self.__is_universal
            else tagged_token
        )

    def tag_sents(self, sentences: list[Sentence]) -> list[TaggedSentence]:
        """جملات را برچسب‌گذاری می‌کند."""
        tagged_sents = super().tag_sents(sentences)
        return (
            [self.__universal_converter(tagged_sent) for tagged_sent in tagged_sents]
            if self.__is_universal
            else tagged_sents
        )


class StanfordPOSTagger(stanford.StanfordPOSTagger):
    """StanfordPOSTagger wrapper."""

    def __init__(
        self,
        model_filename: str,
        path_to_jar: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructor."""
        self._SEPARATOR = "/"
        super().__init__(
            model_filename=model_filename,
            path_to_jar=path_to_jar,
            *args,
            **kwargs,
        )

    def tag(self, tokens: Sentence) -> TaggedSentence:
        """Tag a single sentence."""
        return self.tag_sents([tokens])[0]

    def tag_sents(self, sentences: list[Sentence]) -> list[TaggedSentence]:
        """Tag multiple sentences."""
        refined = ([w.replace(" ", "_") for w in s] for s in sentences)
        return super().tag_sents(list(refined))


class SpacyPOSTagger(POSTagger):
    """Spacy Post Tagger class."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        using_gpu: bool = False,
        gpu_id: int = 0,
    ) -> None:
        """Initialize."""
        super().__init__(universal_tag=True)
        self.model_path = str(model_path) if model_path else None
        self.using_gpu = using_gpu
        self.gpu_id = gpu_id
        self.tagger = None
        self.gpu_availability = False
        
        self.peykare_dict: dict[str, list[str]] = {}
        
        if self.model_path:
             self._setup()

    def _setup(self) -> None:
        """Set up GPU and Load Model."""
        if self.using_gpu:
            self._setup_gpu()
        else:
            logger.info("Using CPU for SpacyPOSTagger.")
        
        if self.model_path and Path(self.model_path).exists():
             self.tagger = spacy.load(self.model_path)
             self.tagger.tokenizer = self._custom_tokenizer

    def _setup_gpu(self) -> None:
        """Check GPU availability."""
        logger.info("GPU Setup Process Started...")
        if spacy.prefer_gpu(gpu_id=self.gpu_id):
            logger.info("GPU is available and ready for use.")
            spacy.require_gpu(gpu_id=self.gpu_id)
            self.gpu_availability = True
        else:
            logger.warning("GPU is not available; spaCy will use CPU.")
            self.gpu_availability = False

    def _custom_tokenizer(self, text: str) -> Doc:
        if self.tagger and text in self.peykare_dict:
            return Doc(self.tagger.vocab, self.peykare_dict[text])
        raise ValueError("No tokenization available for input.")

    def _update_dictionary(self, sents: list[Sentence]) -> None:
        """Add sentences to the custom tokenizer dictionary."""
        for sent in sents:
            key = " ".join(sent)
            if key not in self.peykare_dict:
                self.peykare_dict[key] = sent

    def _setup_dataset(
        self,
        dataset: list[TaggedSentence],
        saved_directory: str,
        data_type: str = "train",
    ) -> None:
        assert data_type in ["train", "test"]
        db = DocBin()
        for sent in tqdm(dataset):
            words = [word for word, _ in sent]
            tags = [tag for _, tag in sent]
            doc = Doc(Vocab(strings=words), words=words)
            for d, tag in zip(doc, tags, strict=True):
                d.tag_ = tag
            db.add(doc)

        path = Path(saved_directory)
        if not path.exists():
            path.mkdir(parents=True)
            
        db.to_disk(f"{saved_directory}/{data_type}.spacy")

    def tag(self, tokens: Sentence, universal_tag: bool = True) -> TaggedSentence:
        """Tag a single sentence."""
        if self.tagger is None:
             raise ValueError("Model is not loaded. Please provide model_path in init.")

        self._update_dictionary([tokens])
        
        text = " ".join(tokens)
        doc = self.tagger(text)
        
        if universal_tag:
            tags = [tok.tag_.replace(",EZ", "") for tok in doc]
        else:
            tags = [tok.tag_ for tok in doc]

        return list(zip(tokens, tags, strict=True))

    def tag_sents(
        self,
        sents: list[Sentence],
        universal_tag: bool = True,
        batch_size: int = 128,
    ) -> list[TaggedSentence]:
        """Tag sentences."""
        if self.tagger is None:
             raise ValueError("Model is not loaded.")

        self._update_dictionary(sents)

        docs = list(
            self.tagger.pipe(
                (" ".join(sent) for sent in sents),
                batch_size=batch_size,
            )
        )
        
        result = []
        for sent, doc in zip(sents, docs, strict=True):
            if universal_tag:
                tags = [tok.tag_.replace(",EZ", "") for tok in doc]
            else:
                tags = [tok.tag_ for tok in doc]
            result.append(list(zip(sent, tags, strict=True)))
            
        return result

    def train(
        self,
        train_dataset: list[TaggedSentence],
        test_dataset: list[TaggedSentence],
        data_directory: str,
        base_config_file: str,
        train_config_path: str,
        output_dir: str,
        use_direct_config: bool = False,
    ) -> None:
        """Train the spaCy model."""
        self.spacy_train_directory = data_directory
        
        if train_dataset:
            self._setup_dataset(
                dataset=train_dataset,
                saved_directory=data_directory,
                data_type="train",
            )

        if test_dataset:
            self._setup_dataset(
                dataset=test_dataset,
                saved_directory=data_directory,
                data_type="test",
            )

        train_data = f"{data_directory}/train.spacy"
        test_data = f"{data_directory}/test.spacy"

        if not use_direct_config:
            logger.info("Setting up training configuration...")
            subprocess.run(
                f"python -m spacy init fill-config {base_config_file} {train_config_path}",
                check=False,
                shell=True
            )
        
        cmd = f"python -m spacy train {train_config_path} --output ./{output_dir} --paths.train ./{train_data} --paths.dev ./{test_data}"
        if self.gpu_availability:
            cmd += f" --gpu-id {self.gpu_id}"

        subprocess.run(cmd, check=False, shell=True)
        self.model_path = f"{output_dir}/model-last"
        
        if test_dataset:
            tokens_list = [[w for w, _ in sent] for sent in test_dataset]
            self._update_dictionary(tokens_list)
            self.tagger = spacy.load(self.model_path)
            self.tagger.tokenizer = self._custom_tokenizer

    def evaluate(self, test_sents: list[TaggedSentence], batch_size: int = 128) -> None:
        """Evaluate the model."""
        tokens_list = [[w for w, _ in sent] for sent in test_sents]
        self._update_dictionary(tokens_list)
        
        if not self.tagger:
            raise ValueError("Model does not exist.")

        gold_labels = [[tag for _, tag in sent] for sent in test_sents]
        prediction_labels = self.tag_sents(tokens_list, batch_size=batch_size, universal_tag=False) # Get raw tags first
        prediction_tags = [[tag for _, tag in sent] for sent in prediction_labels]

        print("-----------------------------------------")
        self._evaluate_tags(gold_labels, prediction_tags, use_ez_tags=True)
        print("-----------------------------------------")
        self._evaluate_tags(gold_labels, prediction_tags, use_ez_tags=False)

    def _evaluate_tags(
        self,
        golds: list[list[str]],
        predictions: list[list[str]],
        use_ez_tags: bool
    ) -> None:
        predictions_cleaned = []
        golds_cleaned = []
        
        def clean_tag(tag: str) -> str:
            if use_ez_tags:
                return "EZ" if "EZ" in tag else "-"
            return tag.replace(",EZ", "")

        for preds, gold_labels in zip(predictions, golds, strict=True):
            for pred in preds:
                predictions_cleaned.append(clean_tag(pred))
            for gold in gold_labels:
                golds_cleaned.append(clean_tag(gold))

        print(classification_report(golds_cleaned, predictions_cleaned))