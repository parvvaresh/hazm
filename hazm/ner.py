import subprocess

import spacy
from spacy.tokens import Doc
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from tqdm import tqdm


class HazmNER:
    """Class for Named Entity Recognition using Hazm and spaCy."""

    def __init__(self, model_path, use_gpu=False) -> None:
        """Initialize the HazmNER object.

        Args:
            model_path: The path to the pre-trained NER model.
            use_gpu: Whether to use GPU for processing.
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = self._load_model(model_path, use_gpu)

    def predict_entities(self, sentences):
        """Predict named entities in a list of sentences.

        Args:
            sentences: List of sentences to predict named entities.

        Returns:
            list of list of tuple: Predicted named entities for each sentence.
        """
        names = []
        for sentence in sentences:
            entities = self.predict_entity(sentence)
            names.append(entities)
        return names

    def predict_entity(self, sentence):
        """Predict named entities in a single sentence.

        Args:
            sentence: Input sentence to predict named entities.

        Returns:
            list of tuple: Predicted named entities in the input sentence.
        """
        doc = self.model(sentence)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def evaluate_model(self, dataset_path):
        """Evaluate the performance of the NER model on a dataset.

        Args:
            dataset_path: Path to the evaluation dataset.
        """
        subprocess.run(f"python -m spacy evaluate {self.model_path} {dataset_path}", check=False)

    def _load_model(self, model_path, use_gpu):
        """Load the trained NER model.

        Args:
            model_path: Path to the trained model.
            use_gpu: Whether to use GPU for processing.

        Returns:
            spacy.Language: Loaded NER model.
        """
        if use_gpu:
            spacy.require_gpu()
        return spacy.load(model_path)
