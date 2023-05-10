---
title: 'langaugemodels: A Python package for learners to explore large language models'
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
 - name: Anderson University
   index: 1
date: 15 June 2023
bibliography: paper.bib
---

# Statement of Need

Large language models are beginning to change how software is developed. The development of the transformer [@vaswani2017attention] has led to rapid progress on many NLP and generative tasks [@bert; @gpt2; @gpt3; @t5; @palm; @flan-t5; @bubeck2023sparks].

Early research indicates that there are many tasks that have been performed by humans that will be transformed by LLMs [@eloundou2023gpts]. For example, large language models that have been trained on code [@codex] are already being used as capable pair programmers via tools such as Microsoft's Copilot. In order build with these technologies in the software of the future, students today must understand their capabilities and begin to learn new paradigms for programming.

The field of machine learning is undergoing a period of rapid progress. Current tools are built on a half-century of growth, discovery, and complex mathematics that are not easily accessible to students beginning to learn the basics of programming. This package seeks to radically lower the barriers to entry on applying these tools to solve problems.

There are many software tools already available for working with large language models. Some of the most popular are :

- Huggingface Transformers [@hftransformers]
- PyTorch [@pytorch]
- TensorFlow [@tensorflow]
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GPT4All](https://github.com/nomic-ai/gpt4all)

While these options serve the needs of researches and hobbyist, they do not provide the cleanest interface for new learners. Some focus on performance over simplicity of installation, and other provide numerous options and error cases that can be painful for inexperienced programmers.

This package use basic types and simple functions while removing the need for opaque boilerplate and configuration options that are not meaningful to new learners.

Let's look at a comparison between Transformers and this package. Here's transformers:

```python
from transformers import pipeline

pipeline(task="text2text-generation",
         model="google/flan-t5-large",
         model_kwargs={"low_cpu_mem_usage": True})

responses = generate("What color is the sky?")
response0 = responses[0]
response0_text = response0["generated_text"]
```

That's not a lot of code, but it does include a lot of magic that could be off-putting to a new learner. In particular:

- `text2text-generation` is a magic string that is meaningless unless you understand the various transformer model architectures
- `google/flan-t5-large` is opaque unless you are familiar with the various models available to the public.
- `model_kwargs={"low_cpu_mem_usage": True}` is especially confusing. Even if you've used `transformers` this may not be familiar. By default, models are loaded in memory then transfered to the inference device (often a GPU). This happens in one large allocation by default. This flag initializes the model in chunks to save CPU memory and allows us to load larger model than we would otherwise be able to when performing CPU inference.
- Unpacking the result is more complicated than necessary. We have a list of dictionaries of results to pull apart to examine the result that we want.

Here's how this works with this package:

```
from languagemodels import lm

response_text = lm.do("What color is the sky?")
```

This intentionally trades flexibility and adaptability for simplicity.

# Implementation

The design of this software package allows its internals to be loosely coupled to the models and inference engines that it uses. At the time of creation, there is rapid progress being made to speed up inference on consumer hardware, but much of this software is difficult to install and may not work easily for all learners.
This package currently uses the HuggingFace Transformers library [@hftransformers] which uses PyTorch [@pytorch] internally for inference.

The current model uses is a variant of the T5 base model [@t5] that has been fine-tuned to better follow instructions [@flan-t5]. As models and inference options become more mature, it will be possible to swap this out with a more powerful that is still able to run on commodity hardware such as Llama [@llama]. In addition to simple local inference, it is also possible to provide API keys to the package to allow access to more powerful hosted inference services.

# References