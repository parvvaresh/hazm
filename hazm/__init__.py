# Base modules (No internal dependencies)
from .types import Token, Tag, TaggedToken, Sentence, TaggedSentence, IOBTag, ChunkedToken, ChunkedSentence
from .constants import *
from .utils import *

# API Protocols
from .api import NormalizerProtocol, TokenizerProtocol, LemmatizerProtocol, TaggerProtocol

# Low-level modules
from .sentence_tokenizer import SentenceTokenizer, sent_tokenize
from .stemmer import Stemmer

# Level 1 modules (depend on utils, constants, api)
from .word_tokenizer import WordTokenizer, word_tokenize

# Level 2 modules (depend on word_tokenizer, stemmer)
from .lemmatizer import Lemmatizer, Conjugation
from .normalizer import Normalizer

# Level 3 modules (depend on lemmatizer, normalizer)
from .token_splitter import TokenSplitter
from .informal_normalizer import InformalNormalizer

# Taggers and Parsers
from .sequence_tagger import SequenceTagger, IOBTagger
from .pos_tagger import POSTagger, SpacyPOSTagger, StanfordPOSTagger
from .chunker import Chunker, RuleBasedChunker, SpacyChunker, tree2brackets
from .dependency_parser import MaltParser

# Embeddings
from .embedding import WordEmbedding, SentEmbedding

# Alias for backward compatibility
DependencyParser = MaltParser


from hazm.corpus_readers import PeykareReader
from hazm.corpus_readers import BijankhanReader
from hazm.corpus_readers import DadeganReader
from hazm.corpus_readers import UniversalDadeganReader
from hazm.corpus_readers import DegarbayanReader
from hazm.corpus_readers import HamshahriReader
from hazm.corpus_readers import MirasTextReader
from hazm.corpus_readers import PersicaReader
from hazm.corpus_readers import QuranReader
from hazm.corpus_readers import SentiPersReader
from hazm.corpus_readers import TNewsReader
from hazm.corpus_readers import TreebankReader
from hazm.corpus_readers import VerbValencyReader
from hazm.corpus_readers import PersianPlainTextReader
from hazm.corpus_readers import WikipediaReader
from hazm.corpus_readers import MizanReader
from hazm.corpus_readers import NerReader
from hazm.corpus_readers import NaabReader
from hazm.corpus_readers import ArmanReader
from hazm.corpus_readers import FaSpellReader
from hazm.corpus_readers import PnSummaryReader