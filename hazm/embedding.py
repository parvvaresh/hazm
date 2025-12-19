"""This module contains classes and functions for converting words or text into numerical vectors."""
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
    """This class includes functions for converting words into numerical vectors."""

    def __init__(self, model: Any, model_type: str) -> None:
        """Constructor.

        Args:
            model: The embedding model.
            model_type: The type of the embedding model.
        """
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
        """Factory method to load the model.

        Args:
            model_path: Path to the model file.
            model_type: Type of the model ('fasttext', 'keyedvector', or 'glove').
            repo_id: Hugging Face repository ID.
            model_filename: Filename in the Hugging Face repository.

        Returns:
            An instance of WordEmbedding.
        """
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
        """Trains the model using Gensim FastText.

        Args:
            dataset_path: Path to the dataset file.
            workers: Number of worker threads.
            vector_size: Dimensionality of the word vectors.
            epochs: Number of epochs to train.
            min_count: The model ignores all words with total frequency lower than this.
            fasttext_type: 'skipgram' or 'cbow'.
            dest_path: Path to save the trained model.
        """
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
        """Returns the vector for the given word.

        Args:
            word: The word to get the vector for.

        Returns:
            The word vector.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model[word]

    def doesnt_match(self, words: list[str]) -> str:
        """Finds the word that does not match the others in the list.

        Args:
            words: A list of words.

        Returns:
            The word that does not match.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.doesnt_match(words)

    def similarity(self, word1: str, word2: str) -> float:
        """Calculates the similarity between two words.

        Args:
            word1: The first word.
            word2: The second word.

        Returns:
            The similarity score.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return float(self.model.similarity(word1, word2))

    def nearest_words(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        """Finds the nearest words to the given word.

        Args:
            word: The word to find nearest neighbors for.
            topn: Number of nearest words to return.

        Returns:
            A list of tuples containing the nearest words and their similarity scores.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.most_similar(word, topn=topn)

    def get_normal_vector(self, word: str) -> ndarray:
        """Returns the normalized vector for the given word.

        Args:
            word: The word to get the normalized vector for.

        Returns:
            The normalized word vector.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.get_vector(word=word, norm=True)

    def get_vocabs(self) -> list[str]:
        """Returns the list of vocabulary words.

        Returns:
            A list of vocabulary words.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.index_to_key

    def get_vocab_to_index(self) -> dict[str, int]:
        """Returns a dictionary mapping words to their indices.

        Returns:
            A dictionary mapping words to indices.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.key_to_index

    def get_vectors(self) -> ndarray:
        """Returns the matrix of word vectors.

        Returns:
            The matrix of word vectors.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.vectors

    def get_vector_size(self) -> int:
        """Returns the size of the word vectors.

        Returns:
            The size of the word vectors.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.vector_size


class SentenceEmbeddingCorpus:
    """Iterate over dataset for Doc2Vec training."""

    def __init__(self, data_path: str) -> None:
        """Constructor.

        Args:
            data_path: Path to the dataset file.
        """
        self.data_path = data_path

    def __iter__(self) -> Iterator[TaggedDocument]:
        """Iterates over the dataset.

        Yields:
            TaggedDocument objects.
        """
        for i, list_of_words in enumerate(smart_open.open(self.data_path, encoding="utf-8")):
            yield TaggedDocument(
                word_tokenize(Normalizer().normalize(list_of_words)),
                [i],
            )


class CallbackSentEmbedding(CallbackAny2Vec):
    """Callback for Doc2Vec training."""

    def __init__(self) -> None:
        """Constructor."""
        self.epoch = 0

    def on_epoch_end(self, model: Doc2Vec) -> None:
        """Called at the end of each epoch.

        Args:
            model: The Doc2Vec model.
        """
        logger.info("Epoch %d of %d...", self.epoch+1, model.epochs)
        self.epoch += 1


class SentEmbedding:
    """Converts sentences to vectors."""

    def __init__(self, model: Doc2Vec | None = None) -> None:
        """Constructor.

        Args:
            model: The Doc2Vec model.
        """
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
        """Factory method to load the model.

        Args:
            model_path: Path to the model file.
            repo_id: Hugging Face repository ID.
            model_filename: Filename in the Hugging Face repository.

        Returns:
            An instance of SentEmbedding.
        """
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
        """Trains the model using Gensim Doc2Vec.

        Args:
            dataset_path: Path to the dataset file.
            min_count: The model ignores all words with total frequency lower than this.
            workers: Number of worker threads.
            windows: The maximum distance between the current and predicted word within a sentence.
            vector_size: Dimensionality of the feature vectors.
            epochs: Number of epochs to train.
            dest_path: Path to save the trained model.
        """
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
        """Returns the vector for the given sentence.

        Args:
            sent: The sentence to get the vector for.

        Returns:
            The sentence vector.
        """
        return self.get_sentence_vector(sent)

    def get_sentence_vector(self, sent: str) -> ndarray:
        """Returns the vector for the given sentence.

        Args:
            sent: The sentence to get the vector for.

        Returns:
            The sentence vector.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        tokenized_sent = word_tokenize(sent)
        return self.model.infer_vector(tokenized_sent)

    def similarity(self, sent1: str, sent2: str) -> float:
        """Calculates the similarity between two sentences.

        Args:
            sent1: The first sentence.
            sent2: The second sentence.

        Returns:
            The similarity score.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return float(self.model.similarity_unseen_docs(
            word_tokenize(sent1),
            word_tokenize(sent2),
        ))

    def get_vector_size(self) -> int:
        """Returns the size of the sentence vectors.

        Returns:
            The size of the sentence vectors.
        """
        if not self.model:
            msg = "Model must be loaded first."
            raise AttributeError(msg)
        return self.model.vector_size
