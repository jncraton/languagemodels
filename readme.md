Language Models
===============

[![PyPI version](https://badge.fury.io/py/languagemodels.svg)](https://badge.fury.io/py/languagemodels)
[![docs](https://img.shields.io/badge/docs-online-brightgreen)](https://languagemodels.netlify.app/)
[![Build](https://github.com/jncraton/languagemodels/actions/workflows/build.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/build.yml)
[![Netlify Status](https://api.netlify.com/api/v1/badges/722e625a-c6bc-4373-bd88-c017adc58c00/deploy-status)](https://app.netlify.com/sites/languagemodels/deploys)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jncraton/languagemodels/blob/master/examples/translate.ipynb)

A Python package providing simple building blocks for exploring natural language processing.

![Translation hello world example](media/hello.gif)

Target Audience
---------------

This package is designed to be as simple as possible for **learners**, **educators**, and **hobbyists** exploring how large language models intersect with modern software development. The interfaces to this package are all simple functions using standard types. The complexity of large language models is hidden from view while providing free local inference using light-weight, open models. All included models are free for educational and commercial use, no API keys are required, and all inference is performed locally by default.

Installation
------------

This package can be installed using the following command:

```sh
pip install languagemodels
```

Example Usage
-------------

Here are some usage examples as Python REPL sessions. This should work in the REPL, notebooks, or in traditional scripts and applications.

### Text Completions

```python
>>> import languagemodels as lm

>>> lm.complete("She hid in her room until")
'she was sure she was safe'
```

### Instruction Following

```python
>>> import languagemodels as lm

>>> lm.do("Translate to English: Hola, mundo!")
'Hello, world!'

>>> lm.do("What is the capital of France?")
'paris'
```

### Classification

```python
>>> import languagemodels as lm

>>> lm.classify("Language models are useful", "positive", "negative")
'positive'
```

### Semantic Search

Semantic search uses an [embedding](https://en.wikipedia.org/wiki/Sentence_embedding) model and vector search under the hood, but that is hidden from view.

```python
>>> import languagemodels as lm

>>> lm.store_doc("Mars is a planet")
>>> lm.store_doc("The sun is hot")
>>> lm.load_doc("What is Mars?")
'Mars is a planet'
```

### Extractive Question Answering

```python
>>> import languagemodels as lm

>>> lm.extract_answer("What color is the ball?", "There is a green ball and a red box")
'green'
```

### External Retrieval

```python
>>> import languagemodels as lm

>>> lm.fetch_wiki('Chemistry')
'Chemistry is the scientific study...

>>> lm.fetch_weather(41.8, -87.6)
'Partly cloudy with a chance of rain...
```

### Misc Text Tools

```python
>>> import languagemodels as lm

>>> get_date()
'Friday, May 12, 2023 at 09:27AM'
```

[Full documentation](https://languagemodels.netlify.app/)

System Requirements
-------------------

This package uses large language models to complete natural language prompts. These models are resource intensive in terms of initial network bandwidth, memory usage, and compute. This package defaults to downloading model weights and computing outputs using the local CPU. The base resource requirements for this are:

|         | Required   | Recommended    |
| ------- | ---------- | -------------- |
| CPU     | Any        | AVX512 support |
| RAM     | 8GB        | 16GB           |
| Storage | 4GB        | 4GB            |

This package will use approximately 5GB of network data to download models initially.

Projects Ideas
--------------

This package can be used to do the heavy lifting for a number of learning projects:

- Basic chatbot
- Chatbot with information retrieval
- Chatbot with access to real-time information
- Tool use
- Text classification
- Extractive question answering
- Semantic search
- Document question answering

Several example programs and notebooks are included in the `examples` directory.