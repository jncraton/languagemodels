Language Models
===============

[![PyPI version](https://badge.fury.io/py/languagemodels.svg)](https://badge.fury.io/py/languagemodels)
[![docs](https://img.shields.io/badge/docs-online-brightgreen)](https://languagemodels.netlify.app/)
[![x64 Build](https://github.com/jncraton/languagemodels/actions/workflows/build.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/build.yml)
[![ARM64 Build](https://github.com/jncraton/languagemodels/actions/workflows/pi.yml/badge.svg)](https://github.com/jncraton/languagemodels/actions/workflows/pi.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jncraton/languagemodels/blob/master/examples/translate.ipynb)

Python building blocks to explore large language models in as little as 512MB of RAM

![Translation hello world example](media/hello.gif)

This package makes using large language models from Python as simple as possible. All inference is performed locally to keep your data private by default.

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

### Instruction Following

```python
>>> import languagemodels as lm

>>> lm.do("Translate to English: Hola, mundo!")
'Hello, world!'

>>> lm.do("What is the capital of France?")
'Paris.'
```

Outputs can be restricted to a list of choices if desired:

```python
>>> lm.do("Is Mars larger than Saturn?", choices=["Yes", "No"])
'No'
```

### Adjusting Model Performance

The base model should run quickly on any system with 512MB of memory, but this memory limit can be increased to select more powerful models that will consume more resources. Here's an example:

```python
>>> import languagemodels as lm
>>> lm.do("If I have 7 apples then eat 5, how many apples do I have?")
'You have 8 apples.'
>>> lm.config["max_ram"] = "4gb"
4.0
>>> lm.do("If I have 7 apples then eat 5, how many apples do I have?")
'I have 2 apples left.'
```

### GPU Acceleration

If you have an NVIDIA GPU with CUDA available, you can opt in to using the GPU for inference:

```python
>>> import languagemodels as lm
>>> lm.config["device"] = "auto"
```

### Text Completions

```python
>>> import languagemodels as lm

>>> lm.complete("She hid in her room until")
'she was sure she was safe'
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

### Code

A model tuned on Python code is included. It can be used to complete code snippets.

```python
>>> import languagemodels as lm
>>> lm.code("""
... a = 2
... b = 5
...
... # Swap a and b
... """)
'a, b = b, a'
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

| Backend                   | Inference Time | Memory Used |
|---------------------------|----------------|-------------|
| Hugging Face transformers | 22s            | 1.77GB      |
| This package              | 11s            | 0.34GB      |

Note that quantization does technically harm output quality slightly, but it should be negligible at this level.

### Models

Sensible default models are provided. The package should improve over time as stronger models become available. The basic models used are 1000x smaller than the largest models in use today. They are useful as learning tools, but perform far below the current state of the art.

Here are the current default models used by the package for a supplied `max_ram` value:

| max_ram | Model Name            | Parameters (B)
| ------- | --------------------- | --------------
| 0.5     | LaMini-Flan-T5-248M   | 0.248
| 1.0     | LaMini-Flan-T5-783M   | 0.783
| 2.0     | LaMini-Flan-T5-783M   | 0.783
| 4.0     | flan-alpaca-gpt4-xl   | 3.0
| 8.0     | openchat-3.5-0106     | 7.0

For code completions, the [CodeT5+](https://arxiv.org/abs/2305.07922) series of models are used.

Commercial Use
--------------

This package itself is licensed for commercial use, but the models used may not be compatible with commercial use. In order to use this package commercially, you can filter models by license type using the `require_model_license` function.

```python
>>> import languagemodels as lm
>>> lm.config['instruct_model']
'LaMini-Flan-T5-248M-ct2-int8'
>>> lm.require_model_license("apache|bsd|mit")
>>> lm.config['instruct_model']
'flan-t5-base-ct2-int8'
```

It is recommended to confirm that the models used meet the licensing requirements for your software.

Projects Ideas
--------------

One of the goals for this package is to be a straightforward tool for learners and educators exploring how large language models intersect with modern software development. It can be used to do the heavy lifting for a number of learning projects:

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
