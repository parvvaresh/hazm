"""این ماژول شامل کلاس‌ها و توابعی برای تبدیل کلمه یا متن به برداری از اعداد است."""
import logging
import multiprocessing
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import smart_open
from gensim.models import Doc2Vec
from gensim.models import FastText
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.fasttext import load_facebook_model
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
    def load(
        cls,
        model_path: str | Path | None = None,
        model_type: str = "fasttext",
        repo_id: str | None = None,
        model_filename: str | None = None,
    ) -> "WordEmbedding":
        """Factory method to load the model."""
        final_model_path = model_path

        if repo_id and model_filename:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub import snapshot_download

                if model_type == "fasttext":
                     final_model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
                else:
                     cache_dir = snapshot_download(repo_id=repo_id)
                     final_model_path = Path(cache_dir) / model_filename

            except ImportError as e:
                msg = f"Failed to import huggingface-hub: {e}"
                raise ImportError(msg) from e
            except Exception as e:
                msg = f"Failed to download from {repo_id}: {e}"
                raise ValueError(msg) from e

        if not final_model_path:
             msg = "Either 'model_path' or 'repo_id' + 'model_filename' must be provided."
             raise ValueError(msg)

        if model_type not in SUPPORTED_EMBEDDINGS:
            msg = f'Model type "{model_type}" is not supported! Choose from {SUPPORTED_EMBEDDINGS}'
            raise KeyError(msg)

        final_model_path = str(final_model_path)
        model = None

        if model_type == "fasttext":
            # Gensim capability to load Facebook's binary format
            try:
                model = load_facebook_model(final_model_path).wv
            except Exception:
                # Fallback: maybe it's a native gensim model
                model = FastText.load(final_model_path).wv

        elif model_type == "keyedvector":
            binary = final_model_path.endswith("bin")
            model = KeyedVectors.load_word2vec_format(final_model_path, binary=binary)

        elif model_type == "glove":
            word2vec_addr = str(final_model_path) + "_word2vec_format.vec"
            if not Path(word2vec_addr).exists():
                logger.info("Converting Glove to Word2Vec format...")
                glove2word2vec(final_model_path, word2vec_addr)
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
        dest_path: str = "fasttext_word2vec_model.model",
    ) -> None:
        """آموزش مدل با استفاده از Gensim FastText."""
        # sg=1 for skipgram, sg=0 for cbow
        sg = 1 if fasttext_type == "skipgram" else 0
        workers = max(1, workers)

        logger.info("Training model with Gensim...")

        corpus = SentenceEmbeddingCorpus(dataset_path)
        sentences = (doc.words for doc in corpus)

        model = FastText(
            vector_size=vector_size,
            window=5,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs,
        )

        model.build_vocab(corpus_iterable=sentences)
        model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=epochs)

        logger.info("Model trained.")
        logger.info("Saving model to %s...", dest_path)
        model.save(dest_path)

        self.model = model.wv

    def __getitem__(self, word: str) -> ndarray:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model[word]

    def doesnt_match(self, words: list[str]) -> str:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.doesnt_match(words)

    def similarity(self, word1: str, word2: str) -> float:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return float(self.model.similarity(word1, word2))

    def nearest_words(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.most_similar(word, topn=topn)

    def get_normal_vector(self, word: str) -> ndarray:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.get_vector(word=word, norm=True)

    def get_vocabs(self) -> list[str]:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.index_to_key

    def get_vocab_to_index(self) -> dict[str, int]:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.key_to_index

    def get_vectors(self) -> ndarray:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.vectors

    def get_vector_size(self) -> int:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.vector_size


class SentenceEmbeddingCorpus:
    """Iterate over dataset for Doc2Vec training."""
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def __iter__(self) -> Iterator[TaggedDocument]:
        for i, list_of_words in enumerate(smart_open.open(self.data_path, encoding="utf-8")):
            yield TaggedDocument(
                word_tokenize(Normalizer().normalize(list_of_words)),
                [i],
            )


class CallbackSentEmbedding(CallbackAny2Vec):
    def __init__(self) -> None:
        self.epoch = 0

    def on_epoch_end(self, model: Doc2Vec) -> None:
        logger.info("Epoch %d of %d...", self.epoch+1, model.epochs)
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
    def load(
        cls,
        model_path: str | Path | None = None,
        repo_id: str | None = None,
        model_filename: str | None = None,
    ) -> "SentEmbedding":

        final_model_path = model_path

        if repo_id and model_filename:
            try:
                from pathlib import Path

                from huggingface_hub import snapshot_download

                cache_dir = snapshot_download(repo_id=repo_id)
                final_model_path = Path(cache_dir) / model_filename

            except ImportError as e:
                msg = f"Failed to download from {repo_id}: {e}"
                raise ImportError(msg) from e
            except Exception as e:
                msg = f"Failed to download from {repo_id}: {e}"
                raise ValueError(msg) from e

        if not final_model_path:
             msg = "Either 'model_path' or 'repo_id' + 'model_filename' must be provided."
             raise ValueError(msg)

        model = Doc2Vec.load(str(final_model_path))
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

        logger.info("Saving model to %s...", dest_path)
        model.save(dest_path)

    def __getitem__(self, sent: str) -> ndarray:
        return self.get_sentence_vector(sent)

    def get_sentence_vector(self, sent: str) -> ndarray:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        tokenized_sent = word_tokenize(sent)
        return self.model.infer_vector(tokenized_sent)

    def similarity(self, sent1: str, sent2: str) -> float:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return float(self.model.similarity_unseen_docs(
            word_tokenize(sent1),
            word_tokenize(sent2),
        ))

    def get_vector_size(self) -> int:
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.vector_size
