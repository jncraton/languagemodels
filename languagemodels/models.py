import os
import re
from huggingface_hub import hf_hub_download
import sentencepiece
from tokenizers import Tokenizer
import ctranslate2


modelcache = {}
max_ram = None
license_match = os.environ.get("LANGUAGEMODELS_MODEL_LICENSE")

# Model list
# This list is sorted in priority order, with the best models first
# The best model that fits in the memory bounds and matches the model filter
# will be selected
models = [
    {
        "name": "flan-alpaca-xl-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "alpaca"],
        "params": 3e9,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",  # HF says apache-2.0, but alpaca is NC
    },
    {
        "name": "flan-alpaca-gpt4-xl-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "gpt4-alpaca"],
        "params": 3e9,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",  # HF says apache-2.0, but alpaca is NC
    },
    {
        "name": "flan-t5-xl-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 3e9,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "LaMini-Flan-T5-783M-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "lamini"],
        "params": 783e6,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-large-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 783e6,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "LaMini-Flan-T5-248M-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "lamini"],
        "params": 248e6,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-alpaca-base-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "alpaca"],
        "params": 248e6,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",  # HF says apache-2.0, but alpaca is NC
    },
    {
        "name": "flan-t5-base-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 248e6,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "LaMini-Flan-T5-77M-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan", "lamini"],
        "params": 77e6,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "cc-by-nc-4.0",
    },
    {
        "name": "flan-t5-small-ct2-int8",
        "tuning": "instruct",
        "datasets": ["c4", "flan"],
        "params": 77e6,
        "quantization": "int8",
        "architecture": "encoder-decoder-transformer",
        "license": "apache-2.0",
    },
    {
        "name": "LaMini-GPT-774M-ct2-int8",
        "tuning": "instruct",
        "datasets": ["webtext", "lamini"],
        "params": 774e6,
        "quantization": "int8",
        "architecture": "decoder-only-transformer",
        "license": "mit",
    },
    {
        "name": "LaMini-GPT-124M-ct2-int8",
        "tuning": "instruct",
        "datasets": ["webtext", "lamini"],
        "params": 124e6,
        "quantization": "int8",
        "architecture": "decoder-only-transformer",
        "license": "mit",
    },
    {
        "name": "all-MiniLM-L6-v2-ct2-int8",
        "tuning": "embedding",
        "params": 22e6,
        "quantization": "int8",
        "architecture": "encoder-only-transformer",
        "license": "apache-2.0",
    },
]


class ModelException(Exception):
    pass


def set_max_ram(value):
    """Sets max allowed RAM

    This value takes priority over environment variables

    Returns the numeric value set in GB

    >>> set_max_ram(16)
    16.0

    >>> set_max_ram('512mb')
    0.5
    """
    global max_ram

    max_ram = convert_to_gb(value)

    return max_ram


def get_max_ram():
    """Return max total RAM to use for models

    max ram will be in GB

    If set_max_ram() has been called, that value will be returned

    Otherwise, value from LANGUAGEMODELS_SIZE env var will be used

    Otherwise, default of 0.45 is returned

    >>> set_max_ram(2)
    2.0

    >>> get_max_ram()
    2.0

    >>> set_max_ram(.5)
    0.5

    >>> get_max_ram()
    0.5
    """

    if max_ram:
        return max_ram

    env = os.environ.get("LANGUAGEMODELS_SIZE")

    if env:
        env = env.lower()

        if env == "small":
            return 0.2
        if env == "base":
            return 0.45
        if env == "large":
            return 1.0
        if env == "xl":
            return 4.0
        if env == "xxl":
            return 16.0

        return convert_to_gb(env)

    return 0.45


def require_model_license(match_re):
    """Require models to match supplied regex

    This can be used to enforce certain licensing constraints when using this
    package.
    """
    global license_match

    license_match = match_re


def convert_to_gb(space):
    """Convert max RAM string to int

    Output will be in gigabytes

    If not specified, input is assumed to be in gigabytes

    >>> convert_to_gb("512")
    512.0

    >>> convert_to_gb(".5")
    0.5

    >>> convert_to_gb("4G")
    4.0

    >>> convert_to_gb("256mb")
    0.25

    >>> convert_to_gb("256M")
    0.25
    """

    if isinstance(space, int) or isinstance(space, float):
        return float(space)

    multipliers = {
        "g": 1.0,
        "m": 2 ** -10,
    }

    space = space.lower()
    space = space.rstrip("b")

    if space[-1] in multipliers:
        return float(space[:-1]) * multipliers[space[-1]]
    else:
        return float(space)


def get_model_name(model_type, max_ram=0.45, license_match=None):
    """Gets an appropriate model name matching current filters

    >>> get_model_name("instruct")
    'LaMini-Flan-T5-248M-ct2-int8'

    >>> get_model_name("instruct", 1.0)
    'LaMini-Flan-T5-783M-ct2-int8'

    >>> get_model_name("instruct", 1.0, "apache*")
    'flan-t5-large-ct2-int8'

    >>> get_model_name("embedding")
    'all-MiniLM-L6-v2-ct2-int8'
    """

    # Allow pinning a specific model via environment variable
    # This is only used for testing
    if os.environ.get("LANGUAGEMODELS_INSTRUCT_MODEL"):
        return os.environ.get("LANGUAGEMODELS_INSTRUCT_MODEL")

    for model in models:
        assert model["quantization"] == "int8"

        memsize = model["params"] / 1e9

        sizefit = memsize < max_ram
        licensematch = not license_match or re.match(license_match, model["license"])

        if model["tuning"] == model_type and sizefit and licensematch:
            return model["name"]

    raise ModelException(f"No valid model found for {model_type}")


def get_model(model_type):
    """Gets a model from the loaded model cache

    >>> tokenizer, model = get_model("instruct")
    >>> type(tokenizer)
    <class 'sentencepiece.SentencePieceProcessor'>

    >>> type(model)
    <class 'ctranslate2._ext.Translator'>

    >>> tokenizer, model = get_model("embedding")
    >>> type(tokenizer)
    <class 'tokenizers.Tokenizer'>

    >>> type(model)
    <class 'ctranslate2._ext.Encoder'>
    """

    model_name = get_model_name(model_type, get_max_ram(), license_match)

    if model_name not in modelcache:
        hf_hub_download(f"jncraton/{model_name}", "config.json")
        model_path = hf_hub_download(f"jncraton/{model_name}", "model.bin")
        model_base_path = model_path[:-10]

        if "minilm" in model_name.lower():
            hf_hub_download(f"jncraton/{model_name}", "vocabulary.txt")
            tokenizer = Tokenizer.from_pretrained(f"jncraton/{model_name}")
            tokenizer.no_padding()
            tokenizer.no_truncation()
            modelcache[model_name] = (
                tokenizer,
                ctranslate2.Encoder(model_base_path),
            )
        elif "gpt" in model_name.lower():
            hf_hub_download(f"jncraton/{model_name}", "vocabulary.json")
            tokenizer = Tokenizer.from_pretrained(f"jncraton/{model_name}")
            # tokenizer.no_padding()
            # tokenizer.no_truncation()
            modelcache[model_name] = (
                tokenizer,
                ctranslate2.Generator(model_base_path),
            )
        else:
            hf_hub_download(f"jncraton/{model_name}", "shared_vocabulary.txt")
            tokenizer_path = hf_hub_download(f"jncraton/{model_name}", "spiece.model")

            tokenizer = sentencepiece.SentencePieceProcessor()
            tokenizer.Load(tokenizer_path)

            modelcache[model_name] = (
                tokenizer,
                ctranslate2.Translator(model_base_path, compute_type="int8"),
            )

    return modelcache[model_name]
