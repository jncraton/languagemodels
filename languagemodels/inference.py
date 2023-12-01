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


def generate_instruct(
    instruction,
    max_tokens=200,
    temperature=0.1,
    topk=1,
    repetition_penalty=1.3,
    prefix="",
    suppress=[],
):
    """Generates one completion for a prompt using an instruction-tuned model

    This may use a local model, or it may make an API call to an external
    model if API keys are available.
    """
    if os.environ.get("LANGUAGEMODELS_TS_KEY") or os.environ.get(
        "LANGUAGEMODELS_TS_SERVER"
    ):
        return generate_ts("flan_t5_xxl_q4", instruction, max_tokens).strip()

    if os.environ.get("LANGUAGEMODELS_OA_KEY"):
        return chat_oa("gpt-3.5-turbo", instruction, max_tokens).strip()

    tokenizer, model = get_model("instruct")

    suppress = [tokenizer.encode(s, add_special_tokens=False).tokens for s in suppress]

    model_info = get_model_info("instruct")

    fmt = model_info.get("prompt_fmt", "{instruction}")

    prompt = fmt.replace("{instruction}", instruction)

    if hasattr(model, "translate_batch"):
        results = model.translate_batch(
            [tokenizer.encode(prompt).tokens],
            target_prefix=[tokenizer.encode(prefix, add_special_tokens=False).tokens],
            repetition_penalty=repetition_penalty,
            max_decoding_length=max_tokens,
            sampling_temperature=temperature,
            sampling_topk=topk,
            suppress_sequences=suppress,
            beam_size=1,
        )
        output_tokens = results[0].hypotheses[0]
        output_ids = [tokenizer.token_to_id(t) for t in output_tokens]
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
    else:
        results = model.generate_batch(
            [tokenizer.encode(prompt).tokens],
            repetition_penalty=repetition_penalty,
            max_length=max_tokens,
            sampling_temperature=temperature,
            sampling_topk=topk,
            suppress_sequences=suppress,
            beam_size=1,
            include_prompt_in_result=False,
        )
        output_ids = results[0].sequences_ids[0]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).lstrip()

    return text


def generate_code(
    prompt,
    max_tokens=200,
    temperature=0.1,
    topk=1,
    repetition_penalty=1.2,
    prefix="",
):
    """Generates one completion for a prompt using a code-tuned model

    >>> generate_code("# Print Hello, World!\\n")
    'print("Hello, World!")\\n'
    """

    tokenizer, model = get_model("code")

    results = model.translate_batch(
        [tokenizer.encode(prompt).tokens],
        target_prefix=[tokenizer.encode(prefix, add_special_tokens=False).tokens],
        repetition_penalty=repetition_penalty,
        max_decoding_length=max_tokens,
        sampling_temperature=temperature,
        sampling_topk=topk,
        beam_size=1,
    )
    output_tokens = results[0].hypotheses[0]
    output_ids = [tokenizer.token_to_id(t) for t in output_tokens]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return text


def rank_instruct(input, targets):
    """Sorts a list of targets by their probabilities

    >>> rank_instruct("Classify positive or negative: I love python. Classification:",
    ... ['positive', 'negative'])
    ['positive', 'negative']

    >>> rank_instruct("Classify fantasy or documentary: "
    ... "The wizard raised their want. Classification:",
    ... ['fantasy', 'documentary'])
    ['fantasy', 'documentary']
    """
    tokenizer, model = get_model("instruct")

    input_tokens = tokenizer.encode(input, add_special_tokens=False).tokens

    if "Generator" in str(type(model)):
        scores = model.score_batch(
            [
                input_tokens + tokenizer.encode(t, add_special_tokens=False).tokens
                for t in targets
            ],
        )
    else:
        scores = model.score_batch(
            [input_tokens] * len(targets),
            target=[
                tokenizer.encode(t, add_special_tokens=False).tokens for t in targets
            ],
        )

    logprobs = [sum(r.log_probs) for r in scores]

    results = sorted(zip(targets, logprobs), key=lambda r: -r[1])

    return [r[0] for r in results]


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
