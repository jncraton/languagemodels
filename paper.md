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

Large language models are having an impact on the way software is designed [@mialon2023augmented]. The development of the transformer [@vaswani2017attention] has led to rapid progress in many NLP and generative tasks [@zhao2023survey; @bert; @gpt2; @gpt3; @t5; @palm; @flan-t5; @bubeck2023sparks]. These models are becoming more powerful as they scale in both parameters [@kaplan2020scaling] and training data [@hoffmann2022training].

Early research suggests that there are many tasks performed by humans that can be transformed by LLMs [@eloundou2023gpts]. For example, large language models trained on code [@codex] are already being used as capable pair programmers via tools such as Microsoft's Copilot. To build with these technologies, students need to understand their capabilities and begin to learn new paradigms for programming.

There are many software tools already available for working with large language models [@hftransformers; @pytorch; @tensorflow; @langchain; @llamacpp; @gpt4all]. While these options serve the needs of software engineers, researchers, and hobbyists, they may not be simple enough for new learners. This package aims to lower the barriers to entry for using these tools in an educational context.

\newpage

# Example Usage

This package eliminates boilerplate and configuration options that create noise for new learners while using only basic types and simple functions. Here's an example from a Python REPL session:

```python
>>> import languagemodels as lm

>>> lm.do("Answer the question: What is the capital of France?")
'Paris.'

>>> lm.do("Classify as positive or negative: I like games",
...       choices=["positive", "negative"])
'positive'

>>> lm.extract_answer("What color is the ball?",
...                   "There is a green ball and a red box")
'green'

>>> lm.get_wiki('Chemistry')
'Chemistry is the scientific study...'

>>> lm.store_doc(lm.get_wiki("Python"), "Python")
>>> lm.store_doc(lm.get_wiki("Javascript"), "Javascript")
>>> lm.get_doc_context("What language is used on the web?")
'From Javascript document: Javascript engines were...'
```

# Features

Despite its simplicity, this package provides a number of building blocks that can be combined to build applications that mimic the architectures of modern software products. Some of the tools included are:

- Instruction following with the `do` function
- Zero-shot classification with the `do` function and `choices` parameter
- Semantic search using the `store_doc` and `get_doc_context` functions
- Extractive question answering using the `extract_answer` function
- Basic web retrieval using the `get_wiki` function

The package includes the following features under the hood:

- Local LLM inference on CPU for broad device support
- Transparent model caching to allow fast repeated inference without explicit model initialization
- Pre-selected models to allow the software to run easily and effectively on as many devices as possible

\newpage

# Implementation

The design of this software package allows its interface to be loosely coupled to the models and inference engines it uses. Progress is being made to speed up inference on consumer hardware, and this package seeks to find a balance between inference efficiency, software stability, and broad hardware support.

This package currently uses CTranslate2 [@ctranslate2] for efficient inference on CPU and GPU. The main models used include Flan-T5 [@flan-t5], LaMini-LM [@lamini-lm], and OpenChat [@openchat]. The default models used by this package can be swapped out in future versions to provide improved generation quality.

# Future work

This package provides a platform for creating simple NLP labs for use in introductory computer science courses. Additional work is needed to design specific learning modules to meet the needs of learners.

Ongoing development efforts will focus on improving the accuracy and efficiency of inference, while keeping the interface stable and supporting all reasonable platforms.

# References
