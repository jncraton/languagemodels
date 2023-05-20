---
title: 'languagemodels: A Python Package for Exploring Modern Natural Language Processing'
tags:
  - Python
  - machine learning
  - language modeling
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

`languagemodels` is a Python package for educators and learners exploring the applications of large language models. It aims to be as easy to set up and use as possible, while providing many of the key building blocks used in modern LLM-driven applications. It is designed to be used in learning modules in introductory programming courses.

# Statement of Need

Large language models are starting to change the way software is designed [@mialon2023augmented]. The development of the transformer [@vaswani2017attention] has led to rapid progress in many NLP and generative tasks [@zhao2023survey; @bert; @gpt2; @gpt3; @t5; @palm; @flan-t5; @bubeck2023sparks]. These models are becoming more powerful as they scale in both parameters [@kaplan2020scaling] and training data [@hoffmann2022training].

Early research suggests that there are many tasks performed by humans that can be transformed by LLMs [@eloundou2023gpts]. For example, large language models trained on code [@codex] are already being used as capable pair programmers via tools such as Microsoft's Copilot. To build with these technologies, students need to understand their capabilities and begin to learn new paradigms for programming.

There are many software tools already available for working with large language models [@hftransformers; @pytorch; @tensorflow; @langchain; @llamacpp; @gpt4all]. While these options serve the needs of software engineers, researchers, and hobbyists, they may not be simple enough for new learners. This package aims to radically lower the barriers to entry for using these tools to solve problems.

\newpage

# Example Usage

This package eliminates boilerplate and configuration options that are meaningless to new learners, and uses basic types and simple functions. Here's an example from a Python REPL session:

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
>>> lm.load_doc("What is Mars?")
'Mars is a planet'

>>> lm.fetch_wiki('Chemistry')
'Chemistry is the scientific study...'
```

# Features

Despite its simplicity, this package provides a number of building blocks that can be combined to build applications that mimic the architectures of modern software products. Some of the tools included are:

- Text generation via the `complete` function
- Instruction following with the `do` function
- Chat-style inference using `chat` function
- Zero-shot classification with the `classify` function
- Semantic search via a document store using the `store_doc` and `load_doc` functions
- Extractive question answering using the `extract_answer` function
- Basic web retrieval using the `fetch_wiki` function

The package includes the following features under the hood

- Local LLM inference on CPU for broad device support
- Transparent model caching to allow fast repeated inference without explicit model initialization
- Pre-selected models to allow the software to run easily and effectively on as many devices as possible

\newpage

# Implementation

The design of this software package allows its internals to be loosely coupled to the models and inference engines it uses. At the time of writing, rapid progress is being made to speed up inference on consumer hardware, but much of this software is difficult to install and may not work well for all learners.

This package currently uses the HuggingFace Transformers library [@hftransformers], which internally uses PyTorch [@pytorch] for inference. The main model used is a variant of the T5 base model [@t5] that has been fine-tuned to better follow instructions [@flan-t5]. Models that focus on inference efficiency are starting to become available [@llama]. It will be possible to replace the internals of this package with more powerful and efficient models in the future. In addition to simple local inference, it is also possible to provide API keys to the package to allow access to more powerful hosted inference services.

# Future work

This package provides a platform for creating simple NLP labs for use in introductory computer science courses. Additional work is needed to design specific learning modules to meet the needs of learners.

Ongoing development efforts will focus on improving the accuracy and efficiency of inference, while keeping the interface stable and supporting all reasonable platforms.

# References