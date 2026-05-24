
# Base modules (No internal dependencies)
from .api import LemmatizerProtocol

# API Protocols
from .api import NormalizerProtocol
from .api import TaggerProtocol
from .api import TokenizerProtocol
from .chunker import Chunker
from .chunker import RuleBasedChunker
from .chunker import SpacyChunker
from .chunker import tree2brackets
from .constants import *
from .dependency_parser import MaltParser
from .dependency_parser import SpacyDependencyParser
from .embedding import SentEmbedding

# Embeddings
from .embedding import WordEmbedding
from .informal_normalizer import InformalNormalizer
from .lemmatizer import Conjugation

# Level 2 modules (depend on word_tokenizer, stemmer)
from .lemmatizer import Lemmatizer
from .normalizer import Normalizer
from .pos_tagger import POSTagger
from .pos_tagger import SpacyPOSTagger
from .pos_tagger import StanfordPOSTagger

# Low-level modules
from .sentence_tokenizer import SentenceTokenizer
from .sentence_tokenizer import sent_tokenize
from .sequence_tagger import IOBTagger

# Taggers and Parsers
from .sequence_tagger import SequenceTagger
from .stemmer import Stemmer

# Level 3 modules (depend on lemmatizer, normalizer)
from .token_splitter import TokenSplitter
from .types import ChunkedSentence
from .types import ChunkedToken
from .types import IOBTag
from .types import Sentence
from .types import Tag
from .types import TaggedSentence
from .types import TaggedToken
from .types import Token
from .utils import *

# Level 1 modules (depend on utils, constants, api)
from .word_tokenizer import WordTokenizer
from .word_tokenizer import word_tokenize

# Alias for backward compatibility
DependencyParser = MaltParser


from hazm.corpus_readers import ArmanReader
from hazm.corpus_readers import BijankhanReader
from hazm.corpus_readers import DadeganReader
from hazm.corpus_readers import DegarbayanReader
from hazm.corpus_readers import FaSpellReader
from hazm.corpus_readers import HamshahriReader
from hazm.corpus_readers import MirasTextReader
from hazm.corpus_readers import MizanReader
from hazm.corpus_readers import NaabReader
from hazm.corpus_readers import NerReader
from hazm.corpus_readers import PersianPlainTextReader
from hazm.corpus_readers import PersicaReader
from hazm.corpus_readers import PeykareReader
from hazm.corpus_readers import PnSummaryReader
from hazm.corpus_readers import QuranReader
from hazm.corpus_readers import SentiPersReader
from hazm.corpus_readers import TNewsReader
from hazm.corpus_readers import TreebankReader
from hazm.corpus_readers import UniversalDadeganReader
from hazm.corpus_readers import VerbValencyReader
from hazm.corpus_readers import WikipediaReader
