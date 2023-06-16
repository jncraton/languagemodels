import os
from huggingface_hub import hf_hub_download
import sentencepiece
from tokenizers import Tokenizer
import ctranslate2


modelcache = {}
max_ram = None


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

    Otherwise, default of 0.5 is returned

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
            return 0.25
        if env == "base":
            return 0.5
        if env == "large":
            return 1.0
        if env == "xl":
            return 4.0
        if env == "xxl":
            return 16.0

        return convert_to_gb(env)

    return 0.5


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
    if model_type == "instruct":
        if get_max_ram() >= 4.0:
            model_name = "jncraton/flan-alpaca-xl-ct2-int8"
        elif get_max_ram() >= 1.0:
            model_name = "jncraton/LaMini-Flan-T5-783M-ct2-int8"
        elif get_max_ram() >= 0.5:
            model_name = "jncraton/LaMini-Flan-T5-248M-ct2-int8"
        else:
            model_name = "jncraton/LaMini-Flan-T5-77M-ct2-int8"
    elif model_type == "embedding":
        model_name = "jncraton/all-MiniLM-L6-v2-ct2-int8"
    else:
        raise ModelException(f"Invalid model: {model_type}")

    if model_name not in modelcache:
        hf_hub_download(model_name, "config.json")
        model_path = hf_hub_download(model_name, "model.bin")
        model_base_path = model_path[:-10]

        if "minilm" in model_name.lower():
            hf_hub_download(model_name, "vocabulary.txt")
            tokenizer = Tokenizer.from_pretrained(model_name)
            tokenizer.no_padding()
            tokenizer.no_truncation()
            modelcache[model_name] = (
                tokenizer,
                ctranslate2.Encoder(model_base_path),
            )
        else:
            hf_hub_download(model_name, "shared_vocabulary.txt")
            tokenizer_path = hf_hub_download(model_name, "spiece.model")

            tokenizer = sentencepiece.SentencePieceProcessor()
            tokenizer.Load(tokenizer_path)

            modelcache[model_name] = (
                tokenizer,
                ctranslate2.Translator(model_base_path, compute_type="int8"),
            )

    return modelcache[model_name]
