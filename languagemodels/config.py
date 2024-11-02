"""Global model and inference configuration

This module manages the global configuration object shared between other
modules in the package. It implements a dictionary with data validation
on the keys and values.

Note that this module provides access to many implementation details
that are not expected to be used by average users. Specific models that
have never been the default for the package may be removed at any time.
"""

import re
import os
from collections import namedtuple
from huggingface_hub import hf_hub_download
import json

ConfigItem = namedtuple("ConfigItem", "initfn default")


class ModelFilterException(Exception):
    pass


# Model list
# This list is sorted in priority order, with the best models first
# The best model that fits in the memory bounds and matches the model filter
# will be selected
models = [
    {
        "name": "openchat-3.5-0106",
        "tuning": "instruct",
        "datasets": ["mistral", "openorca", "flan"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "GPT4 Correct User: {instruction}<|end_of_turn|>" "GPT4 Correct Assistant:"
        ),
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "tuning": "instruct",
        "revision": "d02fc85",
        "datasets": ["llama3"],
        "params": 8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3",
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "Meta-Llama-3-8B-Instruct",
        "tuning": "instruct",
        "datasets": ["llama3"],
        "params": 8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3",
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "openchat-3.5-1210",
        "tuning": "instruct",
        "datasets": ["mistral", "openorca", "flan"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "GPT4 Correct User: {instruction}<|end_of_turn|>" "GPT4 Correct Assistant:"
        ),
    },
    {
        "name": "WizardLM-2-7B",
        "tuning": "instruct",
        "datasets": ["mistral", "wizardlm"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": "USER: {instruction} ASSISTANT:",
    },
    {
        "name": "neural-chat-7b-v3-1",
        "tuning": "instruct",
        "datasets": ["mistral", "slimorca"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "### System:\n"
            "Be helpful\n"
            "### User:\n{instruction}\n"
            "### Assistant:\n"
        ),
    },
    {
        "name": "Mistral-7B-Instruct-v0.2",
        "tuning": "instruct",
        "datasets": ["mistral"],
        "params": 7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": "<s>[INST] {instruction} [/INST]",
    },
    {
        "name": "flan-alpaca-gpt4-xl",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "gpt4-alpaca"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "flan-alpaca-xl",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "alpaca"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "flan-t5-xl",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "Llama-3.2-3B-Instruct",
        "tuning": "instruct",
        "revision": "5da4ba8",
        "datasets": ["llama3"],
        "params": 1e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3.2",
        "repetition_penalty": 1.1,
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "fastchat-t5-3b-v1.0",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "sharegpt"],
        "params": 3e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "LaMini-Flan-T5-783M",
        "tuning": "instruct",
        "revision": "e5e20a1",
        "datasets": ["c4", "flan", "lamini"],
        "params": 783e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-large",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 783e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "Llama-3.2-1B-Instruct",
        "tuning": "instruct",
        "revision": "6e3e3a1",
        "datasets": ["llama3"],
        "params": 1e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "llama3.2",
        "repetition_penalty": 1.1,
        "prompt_fmt": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    {
        "name": "LaMini-Flan-T5-248M",
        "tuning": "instruct",
        "revision": "96cfe99",
        "datasets": ["c4", "flan", "lamini"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-base",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "flan-alpaca-base",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "alpaca"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "dialogstudio-t5-base-v1.0",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "dialogstudio"],
        "params": 248e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
        "prompt_fmt": ("Instruction: Be helpful. <USER> {instruction}"),
    },
    {
        "name": "LaMini-Flan-T5-77M",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "lamini"],
        "params": 77e6,
        "backend": "ct2",
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-small",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 77e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "Phi-3-mini-4k-instruct-20240701",
        "tuning": "instruct",
        "datasets": ["phi-3"],
        "params": 3.8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "mit",
        "prompt_fmt": "<|user|>\n{instruction}<|end|>\n<|assistant|>",
        "repetition_penalty": 1.1,
    },
    {
        "name": "Phi-3-mini-4k-instruct",
        "tuning": "instruct",
        "datasets": ["phi-3"],
        "params": 3.8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "mit",
        "prompt_fmt": "<|user|>\n{instruction}<|end|>\n<|assistant|>",
        "repetition_penalty": 1.1,
    },
    {
        "name": "phi-2",
        "tuning": "instruct",
        "datasets": ["phi-2"],
        "params": 2.7e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "microsoft-research-license",
        "prompt_fmt": "Instruct: {instruction}\nOutput:",
    },
    {
        "name": "gemma-2b-it",
        "tuning": "instruct",
        "datasets": ["gemma"],
        "params": 2.5e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "gemma-terms-of-use",
        "prompt_fmt": "<bos><start_of_turn>user\n"
        "{instruction}<end_of_turn>\n"
        "<start_of_turn>model",
    },
    {
        "name": "h2o-danube3-4b-chat",
        "tuning": "instruct",
        "datasets": [],
        "params": 4.0e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": "<|prompt|>{instruction}</s><|answer|>",
    },
    {
        "name": "h2o-danube2-1.8b-chat",
        "tuning": "instruct",
        "datasets": [],
        "params": 1.8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "other",
        "prompt_fmt": "<|prompt|>{instruction}</s><|answer|>",
    },
    {
        "name": "h2o-danube-1.8b-chat",
        "tuning": "instruct",
        "datasets": [],
        "params": 1.8e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "other",
        "prompt_fmt": "<|prompt|>{instruction}</s><|answer|>",
    },
    {
        "name": "phi-1_5",
        "tuning": "instruct",
        "datasets": ["phi-1_5"],
        "params": 1.4e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "other",
        "prompt_fmt": "{instruction}\n\nAnswer:",
    },
    {
        "name": "h2o-danube3-500m-chat",
        "tuning": "instruct",
        "datasets": [],
        "params": 0.5e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": "<|prompt|>{instruction}</s><|answer|>",
    },
    {
        "name": "SmolLM2-1.7B-Instruct",
        "tuning": "instruct",
        "revision": "83b1658",
        "datasets": [],
        "params": 1.7e9,
        "quantization": "int8",
        "backend": "ct2",
        "context_length": 2048,
        "repetition_penalty": 1.0,
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "<|im_start|>system\nAnswer concisely.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    {
        "name": "SmolLM-1.7B-Instruct",
        "tuning": "instruct",
        "revision": "dc3dfe2",
        "datasets": [],
        "params": 1.7e9,
        "quantization": "int8",
        "backend": "ct2",
        "context_length": 2048,
        "repetition_penalty": 1.1,
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "<|im_start|>system\nAnswer concisely.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    {
        "name": "SmolLM2-360M-Instruct",
        "tuning": "instruct",
        "revision": "ed9c4fe",
        "datasets": [],
        "params": 360e6,
        "quantization": "int8",
        "backend": "ct2",
        "context_length": 2048,
        "repetition_penalty": 1.0,
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "<|im_start|>system\nAnswer concisely.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    {
        "name": "SmolLM-360M-Instruct",
        "tuning": "instruct",
        "revision": "0b0e861",
        "datasets": [],
        "params": 360e6,
        "quantization": "int8",
        "backend": "ct2",
        "context_length": 2048,
        "repetition_penalty": 1.1,
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "<|im_start|>system\nAnswer concisely.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    {
        "name": "SmolLM2-135M-Instruct",
        "tuning": "instruct",
        "revision": "e52a3dc",
        "datasets": [],
        "params": 135e6,
        "quantization": "int8",
        "backend": "ct2",
        "context_length": 2048,
        "repetition_penalty": 1.0,
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "<|im_start|>system\nAnswer concisely.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    {
        "name": "SmolLM-135M-Instruct",
        "tuning": "instruct",
        "revision": "90046ba",
        "datasets": [],
        "params": 135e6,
        "quantization": "int8",
        "backend": "ct2",
        "context_length": 2048,
        "repetition_penalty": 1.3,
        "architecture": "decoder-only-transformer",
        "license": "apache-2.0",
        "prompt_fmt": (
            "<|im_start|>system\nAnswer concisely.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    {
        "name": "LaMini-GPT-774M",
        "tuning": "instruct",
        "datasets": ["webtext", "lamini"],
        "params": 774e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "mit",
        "prompt_fmt": (
            "Below is an instruction that describes a task.\n"
            "Write a response that completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "name": "LaMini-GPT-124M",
        "tuning": "instruct",
        "datasets": ["webtext", "lamini"],
        "params": 124e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "mit",
        "prompt_fmt": (
            "Below is an instruction that describes a task.\n"
            "Write a response that completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "tuning": "instruct",
        "datasets": ["slimpajama", "starcoderdata"],
        "params": 1.1e9,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "decoder-only-transformer",
        "license": "mit",
        "prompt_fmt": ("<|user|>{instruction}<|assistant|>"),
    },
    {
        "name": "codet5p-770m-py",
        "tuning": "code",
        "datasets": ["github-code"],
        "params": 770e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "bsd-3-clause",
    },
    {
        "name": "codet5p-220m-py",
        "tuning": "code",
        "datasets": ["github-code"],
        "params": 220e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-decoder-transformer",
        "license": "bsd-3-clause",
    },
    {
        "name": "all-MiniLM-L6-v2",
        "tuning": "embedding",
        "revision": "28efeb4",
        "params": 22e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-only-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "gte-tiny",
        "tuning": "embedding",
        "params": 22e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-only-transformer",
        "license": "mit",
    },
    {
        "name": "gte-small",
        "tuning": "embedding",
        "params": 33e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-only-transformer",
        "license": "mit",
    },
    {
        "name": "GIST-small-Embedding-v0",
        "tuning": "embedding",
        "params": 33e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-only-transformer",
        "license": "mit",
    },
    {
        "name": "bge-small-en",
        "tuning": "embedding",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "params": 33e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-only-transformer",
        "license": "mit",
    },
    {
        "name": "e5-small-v2",
        "tuning": "embedding",
        "params": 33e6,
        "quantization": "int8",
        "backend": "ct2",
        "architecture": "encoder-only-transformer",
        "license": "mit",
    },
]


class Config(dict):
    """
    Store configuration information for the package.

    This is a dictionary that provides data basic data validation.

    Only appropriate keys and values are allowed to be set.

    >>> c = Config({'max_ram': '4gb'})
    >>> c
    {...'max_ram': 4.0...}

    >>> c = Config({'instruct_model': 'flan-t5-small'})
    >>> c
    {...'instruct_model': 'flan-t5-small'...}

    >>> c = Config({'model_license': 'apache|mit|bsd'})
    >>> c
    {...'model_license': re.compile('apache|mit|bsd')...}

    >>> c = Config({'instruct_model': 'flan-t5-bad'})
    Traceback (most recent call last):
      ...
    KeyError: 'flan-t5-bad'

    >>> c = Config({'bad_value': 1})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_value'

    >>> c = Config()
    >>> c.update({'bad_value': 1})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_value'

    """

    model_names = {m["name"]: m for m in models}

    def __init__(self, config={}):
        # Defaults are loaded first
        for key in Config.schema:
            self[key] = self.schema[key].default

        # Environment variables override defaults
        for key in Config.schema:
            value = os.environ.get(f"LANGUAGEMODELS_{key.upper()}")
            if value:
                self[key] = value

        # Any values passed in the config dict override environment vars
        for key in config.keys():
            self[key] = config[key]

    def __setitem__(self, key, value):
        super().__setitem__(key, Config.schema[key].initfn(value))

        # Auto-adjust instruct_model when filters change
        if key == "max_ram" or key == "model_license":
            found = set()
            for model in models:
                if model["quantization"] == "int8":
                    memsize = model["params"] / 1e9
                elif model["quantization"] == "q3_k_m":
                    memsize = model["params"] * 0.48 / 1e9
                elif model["quantization"] == "q4_k_m":
                    memsize = model["params"] * 0.59 / 1e9

                sizefit = memsize < self["max_ram"]

                if "model_license" in self:
                    licensematch = self["model_license"].match(model["license"])
                else:
                    licensematch = True

                if model["tuning"] not in found and sizefit and licensematch:
                    self[model["tuning"] + "_model"] = model["name"]
                    found.add(model["tuning"])

            if len(found) < 3:
                raise ModelFilterException("Unable to find models to match filters")

    def update(self, other):
        for key in other:
            self[key] = other[key]

    def use_hf_model(self, hf_path, revision, model_type="instruct"):
        """Load and use a model from Huggingface

        :param hf_path: Path for the model e.g. "org/model"
        :param revision: The model git revision to load
        :param model_type: Model type to load
        """

        assert "ct2" in hf_path.lower()
        assert "int8" in hf_path.lower()

        # We defer importing jinja2 until this point as it is only needed
        # for interpolating hf model chat templates and does not need
        # to be installed unless this method is used
        from jinja2 import Environment, BaseLoader

        tok_config = hf_hub_download(
            hf_path, "tokenizer_config.json", revision=revision
        )

        with open(tok_config) as f:
            chat_template = json.load(f)["chat_template"]

        env = Environment(loader=BaseLoader())

        template = env.from_string(chat_template)

        prompt_fmt = template.render(
            messages=[{"role": "user", "content": "{instruction}"}],
            add_generation_prompt=True,
        )

        model = {
            "name": hf_path,
            "backend": "ct2",
            "quantization": "int8",
            "architecture": "decoder-only-transformer",
            "max_tokens": 2048,
            "params": 0,
            "prompt_fmt": prompt_fmt,
        }

        models.insert(0, model)
        self.model_names[model["name"]] = model
        self[f"{model_type}_model"] = model["name"]

    @staticmethod
    def validate_model(model_name):
        return Config.model_names[model_name]["name"]

    @staticmethod
    def validate_device(device):
        assert device in ["auto", "cpu"]

        return device

    @staticmethod
    def convert_to_gb(space):
        """Convert max RAM string to int

        Output will be in gigabytes

        If not specified, input is assumed to be in gigabytes

        >>> Config.convert_to_gb("512")
        512.0

        >>> Config.convert_to_gb(".5")
        0.5

        >>> Config.convert_to_gb("4G")
        4.0

        >>> Config.convert_to_gb("256mb")
        0.25

        >>> Config.convert_to_gb("256M")
        0.25

        >>> Config.convert_to_gb("small")
        0.2

        >>> Config.convert_to_gb("base")
        0.48

        >>> Config.convert_to_gb("large")
        1.0

        >>> Config.convert_to_gb("xl")
        4.0

        >>> Config.convert_to_gb("xxl")
        16.0
        """

        if isinstance(space, int) or isinstance(space, float):
            return float(space)

        size_names = {
            "small": 0.2,
            "base": 0.48,
            "large": 1.0,
            "xl": 4.0,
            "xxl": 16.0,
        }

        if space.lower().strip() in size_names:
            return size_names[space.lower().strip()]

        multipliers = {
            "g": 1.0,
            "m": 2**-10,
        }

        space = space.lower()
        space = space.rstrip("b")

        if space[-1] in multipliers:
            return float(space[:-1]) * multipliers[space[-1]]
        else:
            return float(space)


Config.schema = {
    "max_ram": ConfigItem(Config.convert_to_gb, 0.48),
    "max_tokens": ConfigItem(int, 200),
    "echo": ConfigItem(int, False),
    "device": ConfigItem(Config.validate_device, "cpu"),
    "model_license": ConfigItem(re.compile, ".*"),
    "instruct_model": ConfigItem(Config.validate_model, "LaMini-Flan-T5-248M"),
    "embedding_model": ConfigItem(Config.validate_model, "all-MiniLM-L6-v2"),
    "code_model": ConfigItem(Config.validate_model, "codet5p-220m-py"),
}

config = Config()

if "COLAB_GPU" in os.environ:
    if len(os.environ["COLAB_GPU"]) > 0:
        # We have a Colab GPU, so default to using it
        config["device"] = "auto"
