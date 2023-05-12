Language Models
===============

[![docs](https://img.shields.io/badge/docs-online-brightgreen)](https://languagemodels.netlify.app/)
[![Build](https://github.com/jncraton/languagemodels/actions/workflows/build.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/build.yml)
[![Netlify Status](https://api.netlify.com/api/v1/badges/722e625a-c6bc-4373-bd88-c017adc58c00/deploy-status)](https://app.netlify.com/sites/languagemodels/deploys)

`languagemodels` is a Python package providing simple building blocks for exploring natural language processing.

![Translation hello world example](media/hello.gif)

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

```python
>>> import languagemodels as lm

>>> lm.store_doc("Mars is a planet")
>>> lm.store_doc("The sun is hot")
>>> lm.search_docs("What is Mars?")
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
```

[Full documentation](https://languagemodels.netlify.app/)
