import os
from huggingface_hub import hf_hub_download
import sentencepiece
from tokenizers import Tokenizer
import ctranslate2


modelcache = {}


class ModelException(Exception):
    pass


def convert_mb(space):
    """Convert max RAM string to int

    Output will be in megabytes

    If not specified, input is assumed to be in gigabytes

    >>> convert_mb("512")
    524288

    >>> convert_mb(".5")
    512

    >>> convert_mb("4G")
    4096

    >>> convert_mb("256mb")
    256

    >>> convert_mb("256mb")
    256

    >>> convert_mb("4096kb")
    4
    """

    multipliers = {
        "g": 2**10,
        "m": 1.0,
        "k": 2**-10,
    }

    space = space.lower()
    space = space.rstrip("b")

    if space[-1] in multipliers:
        mb = float(space[:-1]) * multipliers[space[-1]]
    else:
        mb = float(space) * 1024

    return int(mb)


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
        model_name = "jncraton/LaMini-Flan-T5-248M-ct2-int8"
    elif model_type == "embedding":
        model_name = "jncraton/all-MiniLM-L6-v2-ct2-int8"
    else:
        raise ModelException(f"Invalid model: {model_type}")

    if os.environ.get("LANGUAGEMODELS_SIZE") == "small":
        model_name = model_name.replace("base", "small")
        model_name = model_name.replace("248M", "77M")

    if os.environ.get("LANGUAGEMODELS_SIZE") == "large":
        model_name = model_name.replace("base", "large")
        model_name = model_name.replace("248M", "783M")

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
                ctranslate2.Translator(model_base_path),
            )

    return modelcache[model_name]
