import requests
import os
from huggingface_hub import hf_hub_download
import ctranslate2
import re
import sentencepiece


class InferenceException(Exception):
    pass


modelcache = {}


def list_tokens(prompt):
    tokenizer, _ = get_model("jncraton/LaMini-Flan-T5-248M-ct2-int8")

    tokens = tokenizer.EncodeAsPieces(prompt)
    ids = tokenizer.EncodeAsIds(prompt)

    return list(zip(tokens, ids))


def generate_ts(engine, prompt, max_tokens=200):
    """Generates a single text response for a prompt from a textsynth server

    The server and API key are provided as environment variables:

    ts_server is the server such as http://localhost:8080
    ts_key is the API key
    """
    apikey = os.environ.get("ts_key") or ""
    server = os.environ.get("ts_server") or "https://api.textsynth.com"

    response = requests.post(
        f"{server}/v1/engines/{engine}/completions",
        headers={"Authorization": f"Bearer {apikey}"},
        json={"prompt": prompt, "max_tokens": max_tokens},
    )
    resp = response.json()
    if "text" in resp:
        return resp["text"]
    else:
        raise InferenceException(f"TextSynth error: {resp}")


def generate_oa(engine, prompt, max_tokens=200, temperature=0):
    """Generates a single text response for a prompt using OpenAI

    The server and API key are provided as environment variables:

    oa_key is the API key
    """
    apikey = os.environ.get("oa_key")

    response = requests.post(
        "https://api.openai.com/v1/completions",
        headers={
            "Authorization": f"Bearer {apikey}",
            "Content-Type": "application/json",
        },
        json={
            "model": engine,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    resp = response.json()

    try:
        return resp["choices"][0]["text"]
    except KeyError:
        raise InferenceException(f"OpenAI error: {resp}")


def get_model(model_name):
    if os.environ.get("LANGUAGEMODELS_SIZE") == "small":
        model_name = model_name.replace("base", "small")
        model_name = model_name.replace("248M", "77M")

    if os.environ.get("LANGUAGEMODELS_SIZE") == "large":
        model_name = model_name.replace("base", "large")
        model_name = model_name.replace("248M", "783M")

    if model_name not in modelcache:
        hf_hub_download(model_name, "config.json")
        hf_hub_download(model_name, "shared_vocabulary.txt")
        tokenizer_path = hf_hub_download(model_name, "spiece.model")
        model_path = hf_hub_download(model_name, "model.bin")
        model_base_path = model_path[:-10]

        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)

        modelcache[model_name] = (
            tokenizer,
            ctranslate2.Translator(model_base_path),
        )

    return modelcache[model_name]


def generate_instruct(
    prompt,
    max_tokens=200,
    temperature=0.1,
    repetition_penalty=1.2,
    prefix="",
    suppress=[],
):
    """Generates one completion for a prompt using an instruction-tuned model

    This may use a local model, or it may make an API call to an external
    model if API keys are available.
    """
    if os.environ.get("ts_key") or os.environ.get("ts_server"):
        return generate_ts("flan_t5_xxl_q4", prompt, max_tokens)

    if os.environ.get("oa_key"):
        return generate_oa("text-babbage-001", prompt, max_tokens)

    tokenizer, model = get_model("jncraton/LaMini-Flan-T5-248M-ct2-int8")

    suppress = [tokenizer.EncodeAsPieces(s) for s in suppress]

    input_tokens = tokenizer.EncodeAsPieces(prompt) + ["</s>"]
    results = model.translate_batch(
        [input_tokens],
        target_prefix=[tokenizer.EncodeAsPieces(prefix)],
        repetition_penalty=repetition_penalty,
        max_decoding_length=max_tokens,
        sampling_temperature=temperature,
        sampling_topk=40,
        suppress_sequences=suppress,
    )

    output_tokens = results[0].hypotheses[0]

    return tokenizer.DecodePieces(output_tokens)


def parse_chat(prompt):
    """Converts a chat prompt using special tokens to a plain-text prompt

    This is useful for prompting generic models that have not been fine-tuned
    for chat using specialized tokens.

    >>> parse_chat('User: What time is it?')
    Traceback (most recent call last):
        ....
    inference.InferenceException: Chat prompt must end with 'Assistant:'

    >>> parse_chat('''User: What time is it?
    ...
    ...               Assistant:''')
    [{'role': 'user', 'content': 'What time is it?'}]

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: What time is it?
    ...
    ...              Assistant:
    ...              ''')
    [{'role': 'system', 'content': 'A helpful assistant'},
     {'role': 'user', 'content': 'What time is it?'}]

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: What time is it?
    ...
    ...              Assistant: The time is
    ...              ''')
    Traceback (most recent call last):
        ....
    inference.InferenceException: Final assistant message must be blank

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: First para
    ...
    ...              Second para
    ...
    ...              Assistant:
    ...              ''')
    [{'role': 'system', 'content': 'A helpful assistant'},
     {'role': 'user', 'content': 'First para\\n\\nSecond para'}]

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: What time is it?
    ...
    ...              InvalidRole: Nothing
    ...
    ...              Assistant:
    ...              ''')
    Traceback (most recent call last):
        ....
    inference.InferenceException: Invalid chat role: invalidrole
    """

    if not re.match(r"^\s*\w+:", prompt):
        prompt = "System: " + prompt

    prompt = "\n\n" + prompt

    chunks = re.split(r"[\r\n]\s*(\w+):", prompt, flags=re.M)
    chunks = [m.strip() for m in chunks if m.strip()]

    messages = []

    for i in range(0, len(chunks), 2):
        role = chunks[i].lower()

        try:
            content = chunks[i + 1]
            content = re.sub(r"\s*\n\n\s*", "\n\n", content)
        except IndexError:
            content = ""
        messages.append({"role": role, "content": content})

    for message in messages:
        if message["role"] not in ["system", "user", "assistant"]:
            raise InferenceException(f"Invalid chat role: {message['role']}")

    if messages[-1]["role"] != "assistant":
        raise InferenceException("Chat prompt must end with 'Assistant:'")

    if messages[-1]["content"] != "":
        raise InferenceException("Final assistant message must be blank")

    return messages[:-1]
