"""این ماژول شامل کلاس‌ها و توابعی برای تبدیل کلمه یا متن به برداری از اعداد است."""
import multiprocessing
import logging
import warnings
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import fasttext as fstxt
import numpy as np
import smart_open
from gensim.models import Doc2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.scripts.glove2word2vec import glove2word2vec
from numpy import ndarray

from hazm.normalizer import Normalizer
from hazm.word_tokenizer import word_tokenize

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDINGS = ["fasttext", "keyedvector", "glove"]


class WordEmbedding:
    """این کلاس شامل توابعی برای تبدیل کلمه به برداری از اعداد است."""

    def __init__(self, model: Any, model_type: str) -> None:
        self.model = model
        self.model_type = model_type

    @classmethod
    def load(cls, model_path: str | Path, model_type: str) -> "WordEmbedding":
        """Factory method to load the model."""
        if model_type not in SUPPORTED_EMBEDDINGS:
            raise KeyError(f'Model type "{model_type}" is not supported! Choose from {SUPPORTED_EMBEDDINGS}')

        model_path = str(model_path)
        model = None
        
        if model_type == "fasttext":
            model = fstxt.load_facebook_model(model_path).wv
        elif model_type == "keyedvector":
            binary = model_path.endswith("bin")
            model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
        elif model_type == "glove":
            word2vec_addr = str(model_path) + "_word2vec_format.vec"
            if not Path(word2vec_addr).exists():
                logger.info("Converting Glove to Word2Vec format...")
                glove2word2vec(model_path, word2vec_addr)
            model = KeyedVectors.load_word2vec_format(word2vec_addr)
            model_type = "keyedvector"

        return cls(model, model_type)

    def train(
        self,
        dataset_path: str,
        workers: int = multiprocessing.cpu_count() - 1,
        vector_size: int = 200,
        epochs: int = 10,
        min_count: int = 5,
        fasttext_type: str = "skipgram",
        dest_path: str = "fasttext_word2vec_model.bin",
    ) -> None:
        """آموزش مدل (فقط برای FastText در حال حاضر)."""
        if self.model_type != "fasttext":
            warnings.warn(f"Training is only supported for fasttext, not {self.model_type}", stacklevel=2)

        if fasttext_type not in ["cbow", "skipgram"]:
             raise KeyError(f'Invalid fasttext_type "{fasttext_type}"')

        workers = max(1, workers)

        logger.info("Training model...")
        self.model = fstxt.train_unsupervised(
            dataset_path,
            model=fasttext_type,
            dim=vector_size,
            epoch=epochs,
            thread=workers,
            min_count=min_count,
        )
        logger.info("Model trained.")

        logger.info(f"Saving model to {dest_path}...")
        self.model.save_model(dest_path)
        
        self.model = fstxt.load_facebook_model(dest_path).wv

    def __getitem__(self, word: str) -> ndarray:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model[word]

    def doesnt_match(self, words: list[str]) -> str:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.doesnt_match(words)

    def similarity(self, word1: str, word2: str) -> float:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return float(self.model.similarity(word1, word2))

    def nearest_words(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.most_similar(word, topn=topn)

    def get_normal_vector(self, word: str) -> ndarray:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.get_vector(word=word, norm=True)

    def get_vocabs(self) -> list[str]:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.index_to_key

    def get_vocab_to_index(self) -> dict[str, int]:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.key_to_index

    def get_vectors(self) -> ndarray:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.vectors

    def get_vector_size(self) -> int:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.vector_size


class SentenceEmbeddingCorpus:
    """Iterate over dataset for Doc2Vec training."""
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def __iter__(self) -> Iterator[TaggedDocument]:
        for i, list_of_words in enumerate(smart_open.open(self.data_path)):
            yield TaggedDocument(
                word_tokenize(Normalizer().normalize(list_of_words)),
                [i],
            )


class CallbackSentEmbedding(CallbackAny2Vec):
    def __init__(self) -> None:
        self.epoch = 0

    def on_epoch_end(self, model: Doc2Vec) -> None:
        logger.info(f"Epoch {self.epoch+1} of {model.epochs}...")
        self.epoch += 1


class SentEmbedding:
    """تبدیل جمله به بردار."""

    def __init__(self, model: Doc2Vec | None = None) -> None:
        self.model = model
        self.word_embedding: WordEmbedding | None = None
        if self.model:
            self._update_word_embedding()

    def _update_word_embedding(self) -> None:
        if self.model:
            self.word_embedding = WordEmbedding(self.model.wv, "keyedvector")

    @classmethod
    def load(cls, model_path: str) -> "SentEmbedding":
        model = Doc2Vec.load(model_path)
        return cls(model)

    def train(
        self,
        dataset_path: str,
        min_count: int = 5,
        workers: int = multiprocessing.cpu_count() - 1,
        windows: int = 5,
        vector_size: int = 300,
        epochs: int = 10,
        dest_path: str = "gensim_sent2vec.model",
    ) -> None:
        workers = max(1, workers)
        doc = SentenceEmbeddingCorpus(dataset_path)

        logger.info("Initializing Doc2Vec model...")
        model = Doc2Vec(
            min_count=min_count,
            window=windows,
            vector_size=vector_size,
            workers=workers,
        )
        
        logger.info("Building vocab...")
        model.build_vocab(doc)
        
        logger.info("Training model...")
        callbacks = [CallbackSentEmbedding()]
        model.train(doc, total_examples=model.corpus_count, epochs=epochs, callbacks=callbacks)

        model.dv.vectors = np.array([[]]) 
        
        self.model = model
        self._update_word_embedding()
        logger.info("Model trained.")

        logger.info(f"Saving model to {dest_path}...")
        model.save(dest_path)

    def __getitem__(self, sent: str) -> ndarray:
        return self.get_sentence_vector(sent)

    def get_sentence_vector(self, sent: str) -> ndarray:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        tokenized_sent = word_tokenize(sent)
        return self.model.infer_vector(tokenized_sent)

    def similarity(self, sent1: str, sent2: str) -> float:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return float(self.model.similarity_unseen_docs(
            word_tokenize(sent1),
            word_tokenize(sent2),
        ))

    def get_vector_size(self) -> int:
        if not self.model:
            raise AttributeError("Model must be loaded first.")
        return self.model.vector_size