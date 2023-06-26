Language Models
===============

[![PyPI version](https://badge.fury.io/py/languagemodels.svg)](https://badge.fury.io/py/languagemodels)
[![docs](https://img.shields.io/badge/docs-online-brightgreen)](https://languagemodels.netlify.app/)
[![x64 Build](https://github.com/jncraton/languagemodels/actions/workflows/build.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/build.yml)
[![ARM64 Build](https://github.com/jncraton/languagemodels/actions/workflows/pi.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/pi.yml)[![Netlify Status](https://api.netlify.com/api/v1/badges/722e625a-c6bc-4373-bd88-c017adc58c00/deploy-status)](https://app.netlify.com/sites/languagemodels/deploys)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jncraton/languagemodels/blob/master/examples/translate.ipynb)

Python building blocks to explore large language models on any computer with 512MB of RAM

[![Try with Replit Badge](https://replit.com/badge?caption=Try%20with%20Replit&variant=small)](https://replit.com/@jncraton/langaugemodels#main.py)

![Translation hello world example](media/hello.gif)

Target Audience
---------------

This package is designed to be as simple as possible for **learners** and **educators** exploring how large language models intersect with modern software development. The interfaces to this package are all simple functions using standard types. The complexity of large language models is hidden from view while providing free local inference using light-weight, open models. All included models are free for educational use, no API keys are required, and all inference is performed locally by default.

Installation and Getting Started
--------------------------------

This package can be installed using the following command:

```sh
pip install languagemodels
```

Once installed, you should be able to interact with the package in Python as follows:

```python
>>> import languagemodels as lm
>>> lm.do("What color is the sky?")
'The color of the sky is blue.'
```

This will require downloading a significant amount of data (~250MB) on the first run. Models will be cached for later use and subsequent calls should be quick.

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
'Paris.'
```

### Chat

```python
>>> lm.chat('''
...      System: Respond as a helpful assistant.
...
...      User: What time is it?
...
...      Assistant:
...      ''')
'I'm sorry, but as an AI language model, I don't have access to real-time information. Please provide me with the specific time you are asking for so that I can assist you better.'
```

### External Retrieval

Helper functions are provided to retrieve text from external sources that can be used to augment prompt context.

```python
>>> import languagemodels as lm

>>> lm.get_wiki('Chemistry')
'Chemistry is the scientific study...

>>> lm.get_weather(41.8, -87.6)
'Partly cloudy with a chance of rain...

>>> lm.get_date()
'Friday, May 12, 2023 at 09:27AM'
```

Here's an example showing how this can be used (compare to previous chat example):

```python
>>> lm.chat(f'''
...      System: Respond as a helpful assistant. It is {lm.get_date()}
...
...      User: What time is it?
...
...      Assistant:
...      ''')
'It is currently Wednesday, June 07, 2023 at 12:53PM.'
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
>>> lm.store_doc(lm.get_wiki("Python"), "Python")
>>> lm.store_doc(lm.get_wiki("C language"), "C")
>>> lm.store_doc(lm.get_wiki("Javascript"), "Javascript")
>>> lm.get_doc_context("What does it mean for batteries to be included in a language?")
'From Python document: It is often described as a "batteries included" language due to its comprehensive standard library.Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.

From C document: It was designed to be compiled to provide low-level access to memory and language constructs that map efficiently to machine instructions, all with minimal runtime support.'
```

[Full documentation](https://languagemodels.netlify.app/)

### Speed

This package currently outperforms Hugging Face `transformers` for CPU inference thanks to int8 quantization and the [CTranslate2](https://github.com/OpenNMT/CTranslate2) backend. The following table compares CPU inference performance on identical models using the best available quantization on a 20 question test set.

| Backend                  | Inference Time | Memory Used |
|--------------------------|----------------|-------------|
| HuggingFace transformers | 22s            | 1.77GB      |
| This package             | 11s            | 0.34GB      |

Note that quantization does technically harm output quality slightly, but it should be negligible at this level.

### Generation Quality

The models used by this package are 1000x smaller than the largest models in use today. They are useful as learning tools, but if you are expecting ChatGPT or similar performance, you will be very disappointed.

The base model should work on any system with 512MB of memory, but this memory limit can be increased. Setting this value higher will require more memory and generate results more slowly, but the results should be superior. Here's an example:

```python
>>> import languagemodels as lm
>>> lm.do("If I have 7 apples then eat 5, how many apples do I have?")
'You have 8 apples.'
>>> lm.set_max_ram('4gb')
4.0
>>> lm.do("If I have 7 apples then eat 5, how many apples do I have?")
'I have 2 apples left.'
```

This pacakge currently uses [LaMini-Flan-T5-base](https://huggingface.co/MBZUAI/LaMini-Flan-T5-223M) as its default model. This is a fine-tuning of the T5 base model on top of the [FLAN](https://huggingface.co/google/flan-t5-base) fine-tuning provided by Google. The model is tuned to respond to instructions in a human-like manner. The following human evaluations were reported in the [paper](https://github.com/mbzuai-nlp/LaMini-LM) associated with this model family:

![Human-rated model comparison](media/model-comparison.png)

Commercial Use
--------------

This package itself is licensed for commerical use, but the models used may not be compatible with commercial use. In order to use this package commercially, you may want to filter models by their license type using the `require_model_license` function.

```python
>>> import languagemodels as lm
>>> lm.do("What is your favorite animal.")
>>> "As an AI language model, I don't have preferences or emotions."
>>> lm.require_model_license("apache.*|mit")
>>> lm.do("What is your favorite animal.")
'Lion.'
```

The commercially-licensed models may not perform as well as the default models. It is recommended to confirm that the models used do meet the licensing requirements for your software.

Projects Ideas
--------------

This package can be used to do the heavy lifting for a number of learning projects:

- CLI Chatbot (see [examples/chat.py](examples/chat.py))
- Streamlit chatbot (see [examples/streamlitchat.py](examples/streamlitchat.py))
- Chatbot with information retrieval
- Chatbot with access to real-time information
- Tool use
- Text classification
- Extractive question answering
- Semantic search over documents
- Document question answering

Several example programs and notebooks are included in the `examples` directory.
