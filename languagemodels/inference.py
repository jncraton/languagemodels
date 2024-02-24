from typing import List
import requests
import re
import os

from languagemodels.models import get_model, get_model_info


class InferenceException(Exception):
    pass


def list_tokens(prompt):
    """Generates a list of tokens for a supplied prompt

    >>> list_tokens("Hello, world!") # doctest: +SKIP
    [('▁Hello', 8774), (',', 6), ('▁world', 296), ('!', 55)]

    >>> list_tokens("Hello, world!")
    [('...Hello', ...), ... ('...world', ...), ...]
    """
    tokenizer, _ = get_model("instruct")

    output = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = output.tokens
    ids = output.ids

    return list(zip(tokens, ids))


def generate_ts(engine, prompt, max_tokens=200):
    """Generates a single text response for a prompt from a textsynth server

    The server and API key are provided as environment variables:

    LANGUAGEMODELS_TS_SERVER is the server such as http://localhost:8080
    LANGUAGEMODELS_TS_KEY is the API key
    """
    apikey = os.environ.get("LANGUAGEMODELS_TS_KEY") or ""
    server = os.environ.get("LANGUAGEMODELS_TS_SERVER") or "https://api.textsynth.com"

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

    LANGUAGEMODELS_OA_KEY is the API key
    """
    apikey = os.environ.get("LANGUAGEMODELS_OA_KEY")

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


def chat_oa(engine, prompt, max_tokens=200, temperature=0):
    """Generates a single text response for a prompt using OpenAI

    The server and API key are provided as environment variables:

    LANGUAGEMODELS_OA_KEY is the API key
    """
    apikey = os.environ.get("LANGUAGEMODELS_OA_KEY")

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {apikey}",
            "Content-Type": "application/json",
        },
        json={
            "model": engine,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    resp = response.json()

    try:
        return resp["choices"][0]["message"]["content"]
    except KeyError:
        raise InferenceException(f"OpenAI error: {resp}")


def generate(
    instructions: List[str],
    max_tokens: int = 200,
    temperature: float = 0.1,
    topk: int = 1,
    repetition_penalty: float = 1.3,
    prefix: str = "",
    suppress: List[str] = [],
    model: str = "instruct",
):
    """Generates completions for a prompt

    This may use a local model, or it may make an API call to an external
    model if API keys are available.

    >>> generate(["What is the capital of France?"])
    ['...Paris...']
    """
    if os.environ.get("LANGUAGEMODELS_TS_KEY") or os.environ.get(
        "LANGUAGEMODELS_TS_SERVER"
    ):
        return generate_ts("flan_t5_xxl_q4", instructions, max_tokens).strip()

    if os.environ.get("LANGUAGEMODELS_OA_KEY"):
        return chat_oa("gpt-3.5-turbo", instructions, max_tokens).strip()

    tokenizer, model = get_model(model)

    suppress = [tokenizer.encode(s, add_special_tokens=False).tokens for s in suppress]

    model_info = get_model_info("instruct")

    fmt = model_info.get("prompt_fmt", "{instruction}")

    prompts = [fmt.replace("{instruction}", inst) for inst in instructions]

    outputs_ids = []
    if hasattr(model, "translate_batch"):
        prefix = tokenizer.encode(prefix, add_special_tokens=False).tokens
        results = model.translate_batch(
            [tokenizer.encode(p).tokens for p in prompts],
            target_prefix=[prefix] * len(prompts),
            repetition_penalty=repetition_penalty,
            max_decoding_length=max_tokens,
            sampling_temperature=temperature,
            sampling_topk=topk,
            suppress_sequences=suppress,
            beam_size=1,
        )
        outputs_tokens = [r.hypotheses[0] for r in results]
        for output in outputs_tokens:
            outputs_ids.append([tokenizer.token_to_id(t) for t in output])
    else:
        results = model.generate_batch(
            [tokenizer.encode(p).tokens for p in prompts],
            repetition_penalty=repetition_penalty,
            max_length=max_tokens,
            sampling_temperature=temperature,
            sampling_topk=topk,
            suppress_sequences=suppress,
            beam_size=1,
            include_prompt_in_result=False,
        )
        outputs_ids = [r.sequences_ids[0] for r in results]

    return [tokenizer.decode(i, skip_special_tokens=True).lstrip() for i in outputs_ids]


def rank_instruct(inputs, targets):
    """Sorts a list of targets by their probabilities

    >>> rank_instruct(["Classify positive or negative: I love python. Classification:"],
    ... ['positive', 'negative'])
    [['positive', 'negative']]

    >>> rank_instruct(["Classify fantasy or documentary: "
    ... "The wizard raised their wand. Classification:"],
    ... ['fantasy', 'documentary'])
    [['fantasy', 'documentary']]

    >>> rank_instruct(["Say six", "Say seven"], ["six", "seven"])
    [['six', 'seven'], ['seven', 'six']]
    """
    tokenizer, model = get_model("instruct")

    targ_tok = [tokenizer.encode(t).tokens for t in targets]
    targ_tok *= len(inputs)

    in_tok = []
    for input in inputs:
        toks = [tokenizer.encode(input).tokens]
        in_tok += toks * len(targets)

    if "Generator" in str(type(model)):
        scores = model.score_batch([i+t for i, t in zip(in_tok, targ_tok)])
    else:
        scores = model.score_batch(in_tok, target=targ_tok)

    ret = []
    for i in range(0, len(inputs) * len(targets), len(targets)):
        logprobs = [sum(r.log_probs) for r in scores[i:i+len(targets)]]
        results = sorted(zip(targets, logprobs), key=lambda r: -r[1])
        ret.append([r[0] for r in results])

    return ret


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
