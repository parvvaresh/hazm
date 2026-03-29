import os
import shutil
import sys
import urllib
import zipfile
from pathlib import Path

import pytest
from tqdm import tqdm

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
DATA_URL = "https://github.com/roshan-research/hazm-test/archive/refs/heads/main.zip"

class _DownloadProgressBar(tqdm):
    """Progress bar for download using urlretrieve reporthook."""

    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int = -1) -> None:
        if total_size > 0:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def pytest_configure(config):
    """مدیریت هوشمند دانلود و اکسترکت فایل‌های تست."""
    if hasattr(config, "workerinput"):
        return

    if (FILES_DIR / "pos_tagger.model").exists():
        return

    zip_path = BASE_DIR / "test_files.zip"

    if not zip_path.exists():
        print("\n" + "="*50, file=sys.stderr)
        print("Test files not found. Downloading from GitHub...", file=sys.stderr, flush=True)

        try:
            FILES_DIR.mkdir(parents=True, exist_ok=True)
            with _DownloadProgressBar(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc="hazm-test.zip",
                file=sys.stderr,
            ) as progress:
                urllib.request.urlretrieve(DATA_URL, zip_path, reporthook=progress.update_to)
        except Exception as e:
            print(f"\nDownload failed: {e}", file=sys.stderr)
            pytest.exit("Could not download test data.")
    else:
        print("\n" + "="*50, file=sys.stderr)
        print(f"Found existing {zip_path.name}. Skipping download.", file=sys.stderr, flush=True)

    print("Extracting and organizing test files...", file=sys.stderr, flush=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            root_folder_in_zip = next(parts[0] for parts in (name.split("/") for name in zip_ref.namelist()) if parts[0])

            temp_extract_path = BASE_DIR / "temp_extract"
            zip_ref.extractall(temp_extract_path)

            source_dir = temp_extract_path / root_folder_in_zip

            if FILES_DIR.exists():
                shutil.rmtree(FILES_DIR)
            FILES_DIR.mkdir(parents=True, exist_ok=True)

            for item in source_dir.iterdir():
                shutil.move(str(item), str(FILES_DIR))

            shutil.rmtree(temp_extract_path)

        zip_path.unlink()
        print("Done! Test files are ready.\n" + "="*50 + "\n", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"\nExtraction failed: {e}", file=sys.stderr)
        if FILES_DIR.exists():
            shutil.rmtree(FILES_DIR)
        pytest.exit("Test data setup failed.")

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
    return WordEmbedding.load(model_path=str(FILES_DIR / "light_word2vec.bin"), model_type="fasttext")

@pytest.fixture(scope="session")
def sent_embedding():
    return SentEmbedding.load(model_path=str(FILES_DIR / "light_sent2vec.model"))

