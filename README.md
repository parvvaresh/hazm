# Hazm - Persian NLP Toolkit

![Tests](https://img.shields.io/github/actions/workflow/status/roshan-research/hazm/test.yml?branch=master)
![PyPI - Downloads](https://img.shields.io/github/downloads/roshan-research/hazm/total)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hazm)
![GitHub](https://img.shields.io/github/license/roshan-research/hazm)

[**Hazm**](https://www.roshan-ai.ir/hazm/) is a python library to perform natural language processing tasks on Persian text. It offers various features for analyzing, processing, and understanding Persian text. You can use Hazm to normalize text, tokenize sentences and words, lemmatize words, assign part-of-speech tags, identify dependency relations, create word and sentence embeddings, or read popular Persian corpora.

[![sample](documentation/assets/sample.png)](documentation/assets/sample.png)

## Features

- **Normalization:** Converts text to a standard form (diacritics removal, ZWNJ correction, etc).
- **Tokenization:** Splits text into sentences and words.
- **Lemmatization:** Reduces words to their base forms.
- **POS tagging:** Assigns a part of speech to each word.
- **Dependency parsing:** Identifies the syntactic relations between words.
- **Embedding:** Creates vector representations of words and sentences.
- **Hugging Face Integration:** Automatically download and cache pretrained models from the Hub.
- **Persian corpora reading:** Easily read popular Persian corpora with ready-made scripts.

## Installation

To install the latest version of Hazm (requires Python 3.12+), run:

    pip install hazm

To use the pretrained models from Hugging Face, ensure you have the `huggingface-hub` package:

    pip install huggingface-hub

## Pretrained-Models

Hazm supports automatic downloading of pretrained models. You can find all available models (POS Tagger, Chunker, Embeddings, etc.) on our official Hugging Face page:

👉 [**Roshan Research on Hugging Face**](https://huggingface.co/roshan-research/models)

When using Hazm, simply provide the `repo_id` and `model_filename` as shown in the examples below, and the library will handle the rest.

## Usage

```python
from hazm import *

# ===============================
# Stemming
# ===============================
stemmer = Stemmer()
stem = stemmer.stem('کتاب‌ها')
print(stem) # کتاب

# ===============================
# Normalizing
# ===============================
normalizer = Normalizer()
normalized_text = normalizer.normalize('من کتاب های زیــــادی دارم .')
print(normalized_text) # من کتاب‌های زیادی دارم.

# ===============================
# Lemmatizing
# ===============================
lemmatizer = Lemmatizer()
lem = lemmatizer.lemmatize('می‌نویسیم')
print(lem) # نوشت#نویس

# ===============================
# Sentence tokenizing
# ===============================
sentence_tokenizer = SentenceTokenizer()
sent_tokens = sentence_tokenizer.tokenize('ما کتاب می‌خوانیم. یادگیری خوب است.')
print(sent_tokens) # ['ما کتاب می\u200cخوانیم.', 'یادگیری خوب است.']

# ===============================
# Word tokenizing
# ===============================
word_tokenizer = WordTokenizer()
word_tokens = word_tokenizer.tokenize('ما کتاب می‌خوانیم')
print(word_tokens) # ['ما', 'کتاب', 'می\u200cخوانیم']

# ===============================
# Part of speech tagging
# ===============================
tagger = POSTagger(repo_id="roshan-research/hazm-postagger", model_filename="pos_tagger.model")
tagged_words = tagger.tag(word_tokens)
print(tagged_words) # [('ما', 'PRON'), ('کتاب', 'NOUN'), ('می\u200cخوانیم', 'VERB')]

# ===============================
# Chunking
# ===============================
chunker = Chunker(repo_id="roshan-research/hazm-chunker", model_filename="chunker.model")
chunked_tree = tree2brackets(chunker.parse(tagged_words))
print(chunked_tree) # [ما NP] [کتاب NP] [می‌خوانیم VP]

# ===============================
# Word embedding
# ===============================
word_embedding = WordEmbedding.load(repo_id='roshan-research/hazm-word-embedding', model_filename='fasttext_skipgram_300.bin', model_type='fasttext')
odd_word = word_embedding.doesnt_match(['کتاب', 'دفتر', 'قلم', 'پنجره'])
print(odd_word) # پنجره

# ===============================
# Sentence embedding
# ===============================
sent_embedding = SentEmbedding.load(repo_id='roshan-research/hazm-sent-embedding', model_filename='sent2vec-naab.model')
sentence_similarity = sent_embedding.similarity('او شیر میخورد','شیر غذا می‌خورد')
print(sentence_similarity) # 0.4643607437610626

# ===============================
# Dependency parsing
# ===============================
parser = DependencyParser(tagger=tagger, lemmatizer=lemmatizer, repo_id="roshan-research/hazm-dependency-parser", model_filename="langModel.mco")
dependency_graph = parser.parse(word_tokens)
print(dependency_graph)
"""
{0:  {'address': 0,
      'ctag': 'TOP',
      'deps': defaultdict(<class 'list'>, {'root': [3]}),
      'feats': None,
      'head': None,
      'lemma': None,
      'rel': None,
      'tag': 'TOP',
      'word': None},
  1: {'address': 1,
      'ctag': 'PRON',
      'deps': defaultdict(<class 'list'>, {}),
      'feats': '_',
      'head': 3,
      'lemma': 'ما',
      'rel': 'SBJ',
      'tag': 'PRON',
      'word': 'ما'},
  2: {'address': 2,
      'ctag': 'NOUN',
      'deps': defaultdict(<class 'list'>, {}),
      'feats': '_',
      'head': 3,
      'lemma': 'کتاب',
      'rel': 'OBJ',
      'tag': 'NOUN',
      'word': 'کتاب'},
  3: {'address': 3,
      'ctag': 'VERB',
      'deps': defaultdict(<class 'list'>, {'SBJ': [1], 'OBJ': [2]}),
      'feats': '_',
      'head': 0,
      'lemma': 'خواند#خوان',
      'rel': 'root',
      'tag': 'VERB',
      'word': 'می\u200cخوانیم'}})

"""
```

## Documentation

Visit https://roshan-ai.ir/hazm to view the full documentation.

## Evaluation

| Module name      |           |
| :--------------- | --------- |
| DependencyParser | **85.6%** |
| POSTagger        | **98.8%** |
| Chunker          | **93.4%** |
| Lemmatizer       | **89.9%** |

|                                | Metric          | Value   |
| ------------------------------ | --------------- | ------- |
| **SpacyPOSTagger**             | Precision       | 0.99250 |
|                                | Recall          | 0.99249 |
|                                | F1-Score        | 0.99249 |
| **EZ Detection in SpacyPOSTagger** | Precision   | 0.99301 |
|                                | Recall          | 0.99297 |
|                                | F1-Score        | 0.99298 |
| **SpacyChunker**                | Accuracy        | 96.53%  |
|                                | F-Measure       | 95.00%  |
|                                | Recall          | 95.17%  |
|                                | Precision       | 94.83%  |
| **SpacyDependencyParser**       | TOK Accuracy    | 99.06   |
|                                | UAS             | 92.30   |
|                                | LAS             | 89.15   |
|                                | SENT Precision  | 98.84   |
|                                | SENT Recall     | 99.38   |
|                                | SENT F-Measure  | 99.11   |

### Code contributores

![Alt](https://repobeats.axiom.co/api/embed/ae42bda158791645d143c3e3c7f19d8a68d06d08.svg "Repobeats analytics image")

<a href="https://github.com/roshan-research/hazm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=roshan-research/hazm" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=roshan-research/hazm&type=Date)](https://star-history.com/#roshan-research/hazm&Date)

