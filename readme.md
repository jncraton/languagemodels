Language Models
===============

[![docs](https://img.shields.io/badge/docs-online-brightgreen)](https://languagemodels.netlify.app/)
[![Build](https://github.com/jncraton/languagemodels/actions/workflows/build.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/build.yml)
[![Netlify Status](https://api.netlify.com/api/v1/badges/722e625a-c6bc-4373-bd88-c017adc58c00/deploy-status)](https://app.netlify.com/sites/languagemodels/deploys)

Simple building blocks for exploring large language models.

Installation
------------

This package can be installed using the following command:

```sh
pip install languagemodels
```

Example
-------

Here's an example from a Python REPL session:

```python
>>> import languagemodels as lm

>>> lm.chat("What is the capital of France?")
'The capital of France is Paris.'

>>> lm.do("Translate to English: Hola, mundo!")
'Hello, world!'

>>> lm.do("What is the capital of France?")
'paris'

>>> lm.is_positive("Language models are useful")
True

>>> lm.store_doc("Mars is a planet")
>>> lm.store_doc("The sun is hot")
>>> lm.search_docs("What is Mars?")
'Mars is a planet'

>>> lm.search('Chemistry')
'Chemistry is the scientific study...

>>> lm.extract_answer("What color is the ball?", "There is a green ball and a red box")
'green'
```

[Full documentation](https://languagemodels.netlify.app/)
