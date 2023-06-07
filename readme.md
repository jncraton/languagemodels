Language Models
===============

[![PyPI version](https://badge.fury.io/py/languagemodels.svg)](https://badge.fury.io/py/languagemodels)
[![docs](https://img.shields.io/badge/docs-online-brightgreen)](https://languagemodels.netlify.app/)
[![x64 Build](https://github.com/jncraton/languagemodels/actions/workflows/build.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/build.yml)
[![ARM64 Build](https://github.com/jncraton/languagemodels/actions/workflows/pi.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/pi.yml)[![Netlify Status](https://api.netlify.com/api/v1/badges/722e625a-c6bc-4373-bd88-c017adc58c00/deploy-status)](https://app.netlify.com/sites/languagemodels/deploys)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jncraton/languagemodels/blob/master/examples/translate.ipynb)
[![Try with Replit Badge](https://replit.com/badge?caption=Try%20with%20Replit&variant=small)](https://replit.com/@jncraton/langaugemodels#main.py)

A Python package providing simple building blocks to explore large language models on any computer with 512MB of RAM

![Translation hello world example](media/hello.gif)

Target Audience
---------------

This package is designed to be as simple as possible for **learners** and **educators** exploring how large language models intersect with modern software development. The interfaces to this package are all simple functions using standard types. The complexity of large language models is hidden from view while providing free local inference using light-weight, open models. All included models are free for educational use, no API keys are required, and all inference is performed locally by default.

Installation
------------

This package can be installed using the following command:

```sh
pip install languagemodels
```

Model Performance
-----------------

The underlying models used by this package were selected as the best in their size class. These models are 1000x smaller than the largest models in use today. They are useful as learning tools, but if you are expecting ChatGPT or similar performance, you will be very disappointed.

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

>>> lm.get_date()
'Friday, May 12, 2023 at 09:27AM'
```

### Semantic Search

Semantic search is provided to retrieve documents that may provide helpful context from a document store.

```python
>>> import languagemodels as lm

>>> lm.store_doc("Mars is a planet")
>>> lm.store_doc("The sun is hot")
>>> lm.load_doc("What is Mars?")
'Mars is a planet'
```

This can also be used to get a blend of context from stored documents:

```python
>>> import languagemodels as lm

>>> lm.store_doc(lm.fetch_wiki("Python"))
>>> lm.store_doc(lm.fetch_wiki("C++"))
>>> lm.store_doc(lm.fetch_wiki("Javascript"))
>>> lm.store_doc(lm.fetch_wiki("Fortran"))
>>> lm.get_doc_context("What does it mean for batteries to be included in a language?")
'multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language

C, or c, is the third letter in the Latin alphabet, used in the modern English alphabet, the alphabets of other western European languages and others worldwide. Its name in English is cee (pronounced ), plural cees.

a measure of the popularity of programming languages.'
```

[Full documentation](https://languagemodels.netlify.app/)

System Requirements
-------------------

This package uses large language models to complete natural language prompts. These models are resource intensive in terms of initial network bandwidth, memory usage, and compute. This package defaults to downloading model weights and computing outputs using the local CPU. The base resource requirements for this are:

|         | Base (Default) | Recommended    |
| ------- | -------------- | -------------- |
| CPU     | Any            | AVX512 support |
| GPU     | Not required   | Not required   |
| RAM     | 512MB          | 16GB           |
| Storage | 512MB          | 4GB            |

This package will use approximately 5GB of network data to download models initially.

Advanced Usage
--------------

This package is not meant for advanced usage. If you are looking for something more powerful you could explore [transformers](https://huggingface.co/docs/transformers) from Hugging Face.

### Large models

The default model used for inference is around 250M parameters. There is a larger model that can be used if you don't mind things working a little more slowly. It can be enabled by setting the `LANGUAGEMODELS_SIZE` environment variable to `large`. This model isn't large by modern standards and should still work quickly in most environments (but not the lowest tier repl.it instance).

Projects Ideas
--------------

This package can be used to do the heavy lifting for a number of learning projects:

- CLI Chatbot (see examples/chat.py)
- [Streamlit chatbot](https://jncraton-languagemodels-examplesstreamlitchat-g68aa2.streamlit.app/) (see examples/streamlitchat.py)
- Chatbot with information retrieval
- Chatbot with access to real-time information
- Tool use
- Text classification
- Extractive question answering
- Semantic search over documents
- Document question answering

Several example programs and notebooks are included in the `examples` directory.

Attribution
-----------

- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [LaMini-Flan-T5](https://huggingface.co/MBZUAI/LaMini-Flan-T5-783M)
- [Flan-T5](https://huggingface.co/google/flan-t5-large)
