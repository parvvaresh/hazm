import pytest
from pathlib import Path

from hazm import Chunker
from hazm import Conjugation
from hazm import DependencyParser
from hazm import Lemmatizer
from hazm import Normalizer
from hazm import POSTagger
from hazm import RuleBasedChunker
from hazm import SentEmbedding
from hazm import SentenceTokenizer
from hazm import Stemmer
from hazm import TokenSplitter
from hazm import WordEmbedding
from hazm import WordTokenizer

# تعریف مسیر پایه بر اساس مکان فایل conftest.py
BASE_DIR = Path(__file__).parent
FILES_DIR = BASE_DIR / "files"

@pytest.fixture(scope="session")
def stemmer():
    return Stemmer()

@pytest.fixture(scope="session")
def normalizer():
    return Normalizer()

@pytest.fixture(scope="session")
def lemmatizer():
    return Lemmatizer()

@pytest.fixture(scope="session")
def sentence_tokenizer():
    return SentenceTokenizer()

@pytest.fixture(scope="session")
def word_tokenizer():
    return WordTokenizer()

@pytest.fixture(scope="session")
def conjugation():
    return Conjugation()

@pytest.fixture(scope="session")
def pos_tagger():
    return POSTagger(model=str(FILES_DIR / "pos_tagger.model"))

@pytest.fixture(scope="session")
def universal_pos_tagger():
    return POSTagger(model=str(FILES_DIR / "pos_tagger.model"), universal_tag=True)

@pytest.fixture(scope="session")
def token_splitter():
    return TokenSplitter()

@pytest.fixture(scope="session")
def dependency_parser(pos_tagger, lemmatizer):
    return DependencyParser(tagger=pos_tagger, lemmatizer=lemmatizer, working_dir=str(FILES_DIR / "dependency_parser"))

@pytest.fixture(scope="session")
def chunker():
    return Chunker(model=str(FILES_DIR / "chunker.model"))

@pytest.fixture(scope="session")
def rull_based_chunker():
    return RuleBasedChunker()

@pytest.fixture(scope="session")
def word_embedding():
    word_embedding = WordEmbedding(model_type="fasttext")
    word_embedding.load_model(str(FILES_DIR / "light_word2vec.bin"))
    return word_embedding

@pytest.fixture(scope="session")
def sent_embedding():
    sent_embedding = SentEmbedding()
    sent_embedding.load_model(str(FILES_DIR / "light_sent2vec.model"))
    return sent_embedding