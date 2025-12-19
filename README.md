# Hazm - Persian NLP Toolkit

![Tests](https://img.shields.io/github/actions/workflow/status/roshan-research/hazm/test.yml?branch=master)
![PyPI - Downloads](https://img.shields.io/github/downloads/roshan-research/hazm/total)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hazm)
![GitHub](https://img.shields.io/github/license/roshan-research/hazm)

## Introduction

[**Hazm**](https://www.roshan-ai.ir/hazm/) is a python library to perform natural language processing tasks on Persian text. It offers various features for analyzing, processing, and understanding Persian text. You can use Hazm to normalize text, tokenize sentences and words, lemmatize words, assign part-of-speech tags, identify dependency relations, create word and sentence embeddings, or read popular Persian corpora.

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

# --- Normalizer ---
normalizer = Normalizer()
print(normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند'))
# 'اصلاح نویسه‌ها و استفاده از نیم‌فاصله پردازش را آسان می‌کند'

# --- Tokenizer ---
print(sent_tokenize('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟'))
# ['ما هم برای وصل کردن آمدیم!', 'ولی برای پردازش، جدا بهتر نیست؟']
print(word_tokenize('ولی برای پردازش، جدا بهتر نیست؟'))
# ['ولی', 'برای', 'پردازش', '،', 'جدا', 'بهتر', 'نیست', '؟']
# --- Stemmer & Lemmatizer ---
stemmer = Stemmer()
print(stemmer.stem('کتاب‌ها')) # 'کتاب'

lemmatizer = Lemmatizer()
print(lemmatizer.lemmatize('می‌روم')) # 'رفت#رو'

# --- POS Tagger (Automatic download from Hugging Face) ---
tagger = POSTagger(repo_id="roshan-research/hazm-pos-tagger", model_filename="pos_tagger.model")
print(tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم')))
# [('ما', 'PRO'), ('بسیار', 'ADV'), ('کتاب', 'N'), ('می‌خوانیم', 'V')]

# --- Chunker (Automatic download from Hugging Face) ---
chunker = Chunker(repo_id="roshan-research/hazm-chunker", model_filename="chunker.model")
tagged = tagger.tag(word_tokenize('کتاب خواندن را دوست داریم'))
print(tree2brackets(chunker.parse(tagged)))
# '[کتاب خواندن NP] [را POSTP] [دوست داریم VP]'

# --- Word Embedding (Automatic download from Hugging Face) ---
word_embedding = WordEmbedding.load(repo_id='roshan-research/hazm-word-embedding', model_filename='fasttext_skipgram_300.bin', model_type='fasttext')
print(word_embedding.doesnt_match(['سلام', 'درود', 'خداحافظ', 'پنجره'])) # 'پنجره'

# --- Sent Embedding (Automatic download from Hugging Face) ---
sent_embedding = SentEmbedding.load(repo_id='roshan-research/hazm-sent-embedding', model_filename='sent2vec-naab.model')
vector = sent_embedding.get_sentence_vector('این یک جمله نمونه برای تبدیل به بردار است.')
print(sent_embedding.similarity('شیر حیوانی وحشی است', 'پلنگ از دیگر جانوران درنده است'))
# 0.85 (مثلا)

# --- Dependency Parser ---
parser = DependencyParser(tagger=tagger, lemmatizer=lemmatizer, repo_id="roshan-research/hazm-dependency-parser", model_filename="langModel.mco")
graph = parser.parse(word_tokenize('زنگ‌ها برای که به صدا درمی‌آید؟'))
print(graph)
```

## Documentation

Visit https://roshan-ai.ir/hazm/docs to view the full documentation.

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
