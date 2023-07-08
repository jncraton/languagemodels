import re
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import ctranslate2

from languagemodels.config import config, models


modelcache = {}


class ModelException(Exception):
    pass


def get_model_info(model_type="instruct", max_ram=None, license_match=None):
    """Gets info about the current model in use

    >>> get_model_info('instruct')
    {'name': 'LaMini-Flan-T5-248M-ct2-int8', 'tuning': 'instruct'...
    """
    if not max_ram:
        max_ram = config["max_ram"]

    model_name = get_model_name(model_type, max_ram, license_match)

    m = [m for m in models if m["name"] == model_name][0]

    param_bits = int(re.search(r"\d+", m["quantization"]).group(0))

    m["size_gb"] = m["params"] * param_bits / 8 / 1e9

    return m


def get_model_name(model_type, max_ram=0.40, license_match=None):
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
    if os.environ.get("LANGUAGEMODELS_INSTRUCT_MODEL") and model_type == "instruct":
        return os.environ.get("LANGUAGEMODELS_INSTRUCT_MODEL")

    for model in models:
        assert model["quantization"] == "int8"

        memsize = model["params"] / 1e9

        sizefit = memsize < max_ram
        licensematch = not license_match or re.match(license_match, model["license"])

        if model["tuning"] == model_type and sizefit and licensematch:
            return model["name"]

    raise ModelException(f"No valid model found for {model_type}")


def get_model(model_type, tokenizer_only=False):
    """Gets a model from the loaded model cache

    If tokenizer_only, the model itself will not be (re)loaded

    >>> tokenizer, model = get_model("instruct")
    >>> type(tokenizer)
    <class 'tokenizers.Tokenizer'>

    >>> type(model)
    <class 'ctranslate2._ext.Translator'>

    >>> tokenizer, model = get_model("embedding")
    >>> type(tokenizer)
    <class 'tokenizers.Tokenizer'>

    >>> type(model)
    <class 'ctranslate2._ext.Encoder'>
    """

    model_name = get_model_name(model_type, config["max_ram"], config["model_license"])

    if config["max_ram"] < 4 and not tokenizer_only:
        for model in modelcache:
            if model != model_name:
                try:
                    modelcache[model][1].unload_model()
                except AttributeError:
                    # Encoder-only models can't be unloaded by ctranslate2
                    pass

    if model_name not in modelcache:
        model = None

        hf_hub_download(f"jncraton/{model_name}", "config.json")
        model_path = hf_hub_download(f"jncraton/{model_name}", "model.bin")
        model_base_path = model_path[:-10]
        tok_config = hf_hub_download(f"jncraton/{model_name}", "tokenizer.json")
        tokenizer = Tokenizer.from_file(tok_config)

        if "minilm" in model_name.lower():
            tokenizer.no_padding()
            tokenizer.no_truncation()

            if not tokenizer_only:
                hf_hub_download(f"jncraton/{model_name}", "vocabulary.txt")
                model = ctranslate2.Encoder(model_base_path, compute_type="int8")

            modelcache[model_name] = (
                tokenizer,
                model,
            )
        elif "gpt" in model_name.lower():
            if not tokenizer_only:
                hf_hub_download(f"jncraton/{model_name}", "vocabulary.json")
                model = ctranslate2.Generator(model_base_path, compute_type="int8")
            modelcache[model_name] = (
                tokenizer,
                model,
            )
        else:
            if not tokenizer_only:
                hf_hub_download(f"jncraton/{model_name}", "shared_vocabulary.txt")
                model = ctranslate2.Translator(model_base_path, compute_type="int8")
            modelcache[model_name] = (
                tokenizer,
                model,
            )
    elif not tokenizer_only:
        # Make sure the model is reloaded if we've unloaded it
        try:
            modelcache[model_name][1].load_model()
        except AttributeError:
            # Encoder-only models can't be unloaded in ctranslate2
            pass

    return modelcache[model_name]
