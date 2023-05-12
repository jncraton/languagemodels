---
title: 'langaugemodels: A Python Package for Learners Exploring Natural Language Process'
tags:
  - Python
  - machine learning
  - language modelling
  - nlp
authors:
  - name: Jonathan L. Craton
    orcid: 0009-0007-6543-8571
    affiliation: 1
affiliations:
 - name: Department of Computer Science, Anderson University (IN)
   index: 1
date: 15 June 2023
bibliography: paper.bib
---

# Summary

`languagemodels` is a Python package for educators and learners exploring the applications of large language models. It strives to be as simple as possible to setup and use while providing many of the key building blocks used in modern LLM-driven applications.

This package should work in most Python environments. The following is the most trivial example that will work in notebook environments.

```python
%pip install languagemodels

import languagemodels as lm

lm.complete("Once there was a")
```

# Statement of Need

Large language models are beginning to change how software is designed. The development of the transformer [@vaswani2017attention] has led to rapid progress on many NLP and generative tasks [@bert; @gpt2; @gpt3; @t5; @palm; @flan-t5; @bubeck2023sparks]. These models continue to become more powerful as they scale up in both parameters [@kaplan2020scaling] as well as training data [@hoffmann2022training].

Early research indicates that there are many tasks that have been performed by humans that can be transformed by LLMs [@eloundou2023gpts]. For example, large language models that have been trained on code [@codex] are already being used as capable pair programmers via tools such as Microsoft's Copilot. In order build with these technologies, students must understand their capabilities and begin to learn new paradigms for programming.

There are many software tools already available for working with large language models [@hftransformers; @pytorch; @tensorflow; @langchain; @llamacpp; @gpt4all]. While these options serve the needs of software engineers, researches, and hobbyists, they may not be simple enough for new learners. This package seeks to radically lower the barriers to entry when applying these tools to solve problems.

# Example Usage

This package uses basic types and simple functions while removing the need for opaque boilerplate and configuration options that are not meaningful to new learners. Here's an example from a Python REPL session:

```python
>>> import languagemodels as lm

>>> lm.complete("She hid in her room until")
'she was sure she was safe'

>>> lm.do("Translate to English: Hola, mundo!")
'Hello, world!'

>>> lm.do("What is the capital of France?")
'paris'

>>> lm.classify("Language models are useful", "positive", "negative")
'positive'

>>> lm.extract_answer("What color is the ball?", "There is a green ball and a red box")
'green'

>>> lm.store_doc("Mars is a planet")
>>> lm.store_doc("The sun is hot")
>>> lm.search_docs("What is Mars?")
'Mars is a planet'

>>> lm.fetch_wiki('Chemistry')
'Chemistry is the scientific study...'
```

# Features

Despite its simplicity, this package provides a number of building blocks that can be composed to build applications that mimic the architectures of contemporary tools such as Phind or Bing Chat. Some of the included tools are:

- Text generation via the `complete` function
- Instruction following via the `do` function
- Zero-shot classification using the `classify` function
- Semantic search via a document store using the `store_doc` and `search_docs` functions.
- Extractive question answering via the `extract_answer` function
- Basic web retrieval using the `fetch_wiki` function

In order make this as simple as possible, the package includes the following features under the hood:

- Local LLM inference via `transformers`
- Transparent model caching to allow fast repeated inference with explicit model initialization
- Opinionated model selections to allow the software to run easily and effectively on as many devices as possible
- Local document embedding and vector search

# Implementation

The design of this software package allows its internals to be loosely coupled to the models and inference engines that it uses. At the time of creation, there is rapid progress being made to speed up inference on consumer hardware, but much of this software is difficult to install and may not work easily for all learners.
This package currently uses the HuggingFace Transformers library [@hftransformers] which uses PyTorch [@pytorch] internally for inference.

The current model uses is a variant of the T5 base model [@t5] that has been fine-tuned to better follow instructions [@flan-t5]. As models and inference options become more mature, it will be possible to swap this out with a more powerful that is still able to run on commodity hardware such as Llama [@llama]. In addition to simple local inference, it is also possible to provide API keys to the package to allow access to more powerful hosted inference services.

# References