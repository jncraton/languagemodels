from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

modelcache = {}
tokenizercache = {}


def get_model(model):
    if model not in modelcache:
        modelcache[model] = AutoModelForSeq2SeqLM.from_pretrained(
            model, low_cpu_mem_usage=True
        )

    return modelcache[model]


def get_tokenizer(tokenizer):
    if tokenizer not in tokenizercache:
        tokenizercache[tokenizer] = AutoTokenizer.from_pretrained(tokenizer)

    return tokenizercache[tokenizer]
