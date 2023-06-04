import requests
import os
from transformers import pipeline, T5Tokenizer
from huggingface_hub import snapshot_download
import ctranslate2
import re
import sentencepiece


class InferenceException(Exception):
    pass


modelcache = {}


def list_tokens(prompt):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    )

    ids = [int(id) for id in input_ids[0]]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True)

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


def generate_instruct(prompt, max_tokens=200, temperature=0.1, repetition_penalty=1.2):
    """Generates one completion for a prompt using an instruction-tuned model

    This may use a local model, or it may make an API call to an external
    model if API keys are available.
    """
    if os.environ.get("ts_key") or os.environ.get("ts_server"):
        return generate_ts("flan_t5_xxl_q4", prompt, max_tokens)

    if os.environ.get("oa_key"):
        return generate_oa("text-babbage-001", prompt, max_tokens)

    if "test" not in modelcache:
        model_path = snapshot_download(repo_id="jncraton/flan-t5-large-ct2-int8")
        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.Load(f"{model_path}/spiece.model")

        modelcache["test"] = (
            tokenizer,
            ctranslate2.Translator(model_path),
        )

    tokenizer, model = modelcache["test"]

    input_tokens = tokenizer.EncodeAsPieces(prompt) + ["</s>"]
    results = model.translate_batch([input_tokens])

    output_tokens = results[0].hypotheses[0]

    return tokenizer.DecodePieces(output_tokens)


def get_pipeline(task, model):
    """Gets a pipeline instance

    This is thin wrapper around the pipeline constructor to provide caching
    across calls.
    """

    if model not in modelcache:
        modelcache[model] = pipeline(
            task, model=model, model_kwargs={"low_cpu_mem_usage": True}
        )

    return modelcache[model]


def convert_chat(prompt):
    """Converts a chat prompt using special tokens to a plain-text prompt

    This is useful for prompting generic models that have not been fine-tuned
    for chat using specialized tokens.

    >>> convert_chat("<|system|>A helpful assistant<|endoftext|>" \\
    ...              "<|prompter|>What time is it?<|endoftext|>" \\
    ...              "<|assistant|>")
    'A helpful assistant\\n\\nUser:What time is it?\\n\\nAssistant:'

    >>> convert_chat("<|prompter|>Who are you?<|endoftext|>" \\
    ...              "<|assistant|>")
    'User:Who are you?\\n\\nAssistant:'

    >>> convert_chat("<|prompter|>What is 1+1?<|endoftext|>\\n\\n" \\
    ...              "<|assistant|>")
    'User:What is 1+1?\\n\\nAssistant:'

    >>> convert_chat("<|system|>A friend<|endoftext|>" \\
    ...              "<|prompter|>Hi<|endoftext|>" \\
    ...              "<|assistant|>Yo<|endoftext|>" \\
    ...              "<|prompter|>We good?<|endoftext|>" \\
    ...              "<|assistant|>")
    'A friend\\n\\nUser:Hi\\n\\nAssistant:Yo\\n\\nUser:We good?\\n\\nAssistant:'
    >>> convert_chat("\\n<|system|>Be nice<|endoftext|>" \\
    ...              "<|prompter|>brb\\n<|endoftext|>" \\
    ...              "<|assistant|>k<|endoftext|>" \\
    ...              "<|prompter|>back<|endoftext|>" \\
    ...              "<|assistant|>")
    'Be nice\\n\\nUser:brb\\n\\nAssistant:k\\n\\nUser:back\\n\\nAssistant:'

    >>> convert_chat("<|user|>Who are you?<|endoftext|>" \\
    ...              "<|assistant|>")
    Traceback (most recent call last):
        ....
    inference.InferenceException: Invalid special token in chat prompt: <|user|>
    """

    prompt = re.sub(r"\s*<\|system\|>\s*", "", prompt)
    prompt = re.sub(r"\s*<\|prompter\|>\s*", "User:", prompt)
    prompt = re.sub(r"\s*<\|assistant\|>\s*", "Assistant:", prompt)
    prompt = re.sub(r"\s*<\|endoftext\|>\s*", "\n\n", prompt)

    special_token_match = re.search(r"<\|.*?\|>", prompt)
    if special_token_match:
        token_text = special_token_match.group(0)
        raise InferenceException(f"Invalid special token in chat prompt: {token_text}")

    return prompt
