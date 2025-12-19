# Hazm: Persian NLP Toolkit

[Hazm](https://www.roshan-ai.ir/hazm) is a **comprehensive Python library for processing the Persian language**. It provides tools for text normalization, sentence and word tokenization, stemming, lemmatization, part-of-speech tagging, syntactic dependency parsing, and more.

!!! info "Compatible with Python {{needed_python_version()}}"
    Hazm is built on top of the [NLTK](https://www.nltk.org/) library and is specifically optimized for the Persian language. It is fully compatible with Python {{needed_python_version()}}.

!!! info "Maintained by Roshan"
    Originally started as a personal project, Hazm is now developed and maintained under the [Roshan AI](https://www.roshan-ai.ir/) team.

<figure markdown>
  ![Hazm library](assets/sample.png){ loading=lazy }
  <figcaption>Persian Natural Language Processing made easy with Hazm</figcaption>
</figure>

## Installation

You can install Hazm using pip:

```console
$ pip install hazm
```

## Pretrained Models

Hazm requires pretrained models for advanced tasks such as POS tagging, Chunking, and Dependency Parsing. There are two ways to use these models:

### 1. Automatic Loading (Hugging Face Hub)
The latest version of Hazm integrates directly with the **Hugging Face Hub**. You can load models automatically by providing the `repo_id` and `model_filename`:

```python
from hazm import POSTagger

# This will automatically download and cache the model from Hugging Face
tagger = POSTagger(
    repo_id="roshan-research/hazm-postagger", 
    model_filename="pos_tagger.model"
)
```

### 2. Manual Loading
If you prefer to work offline, you can [download the models]({{pretrained_models}}) manually and provide the local path to the constructor:

```python
from hazm import POSTagger

# Provide the local path to the downloaded model file
tagger = POSTagger(model="path/to/your/pos_tagger.model")
```

## Quick Start

Import Hazm into your project and start processing Persian text immediately:

```python
from hazm import *
```

{{hazm_code_example()}}

## Next Steps

*   Explore detailed documentation for each module in the **[Classes and Functions](content/hazm/index.md)** section.
*   Learn how to work with various Persian corpora in the **[Corpus Readers](content/hazm/corpus_readers/index.md)** section.
*   If you are looking for Hazm in other environments, check out the [Ports in Other Languages](content/in-other-languages.md) section.