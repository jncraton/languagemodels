Language Models
===============

[![Build](https://github.com/jncraton/languagemodels/actions/workflows/build.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/build.yml)

Simple building blocks for exploring large language models.

Example
-------

Here's an example from a Python REPL session:

```python
>>> import languagemodels as lm

>>> lm.chat("What is the capital of France?")
'The capital of France is Paris.'

>>> lm.do("Translate to English: Hola, mundo!")
'Hello, world!'

>>> lm.is_positive("Language models are useful")
True

>>> lm.match("What is Mars?", ["Mars is a planet", "The sun is hot"])
'Mars is a planet'

>>> lm.search('Chemistry')
'Chemistry is the scientific study...

>>> lm.extract_answer("What color is the ball?", "There is a green ball and a red box")
'green'
```
