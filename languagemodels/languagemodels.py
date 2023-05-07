from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

modelcache = {}
tokenizercache = {}


def get_model(model):
    if model not in modelcache:
        modelcache[model] = AutoModelForSeq2SeqLM.from_pretrained(model)

    return modelcache[model]


def get_tokenizer(tokenizer):
    if tokenizer not in tokenizercache:
        tokenizercache[tokenizer] = AutoTokenizer.from_pretrained(tokenizer)

    return tokenizercache[tokenizer]


def chat(userprompt):
    model = get_model("google/flan-t5-base")
    tokenizer = get_tokenizer("google/flan-t5-base")

    prompt = (
        f"System: "
        f"Agent responses will be truthful, helpful, and harmless.\n"
        f"User: {userprompt}\n"
        f"Agent: "
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.2)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
