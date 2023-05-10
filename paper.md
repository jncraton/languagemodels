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

There are many software tools already available for working with large language models. Some of the most popular are :

- Huggingface Transformers [@hftransformers]
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GPT4All](https://github.com/nomic-ai/gpt4all)

While these options serve the needs of researches and hobbyist, they do not provide the cleanest interface for new learners. Some focus on performance over simplicity of installation, and other provide numerous options and error cases that can be painful for inexperienced programmers.

This package seeks to be as simple as possible so that someone with no experience in the field of computer science could pick up these tools and begin to experiment with them. The following are the goals for the package:

- Use simple types. Where possible, built-in types such as strings, Booleans, and numbers should be favored. Even lists and dictionaries should be avoided to keep the interface as simple as possible for new programmers.
- Avoid classes.

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

# References