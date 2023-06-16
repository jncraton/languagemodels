import os
from huggingface_hub import hf_hub_download
import sentencepiece
from tokenizers import Tokenizer
import ctranslate2


modelcache = {}


class ModelException(Exception):
    pass


def get_model(model_type):
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
