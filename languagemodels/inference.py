from typing import List
import sys
from time import perf_counter

from languagemodels.models import get_model, get_model_info
from languagemodels.config import config


class InferenceException(Exception):
    pass


def truncate_prompt(prompt):
    """Truncates a prompt to the maximum length allowed by the config"""
    max_prompt_length = config["max_prompt_length"]
    if len(prompt) > max_prompt_length:
        print(
            f"Warning: Prompt truncated from {len(prompt)} to "
            f"{max_prompt_length} characters to avoid OOM."
        )
        return prompt[:max_prompt_length]
    return prompt


def list_tokens(prompt):
    """Generates a list of tokens for a supplied prompt

    >>> list_tokens("Hello, world!") # doctest: +SKIP
    [('▁Hello', 8774), (',', 6), ('▁world', 296), ('!', 55)]

    >>> list_tokens("Hello, world!")
    [('...Hello', ...), ... ('...world', ...), ...]
    """
    prompt = truncate_prompt(prompt)
    tokenizer, _ = get_model("instruct")

    output = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = output.tokens
    ids = output.ids

    return list(zip(tokens, ids))


def stream_results(results, tokenizer):
    """Map a token iterator to a substring iterator"""
    tokens = []
    last_len = 0

    for result in results:
        tokens.append(result.token_id)
        text = tokenizer.decode(tokens)
        yield text[last_len:]
        last_len = len(text)


def echo_results(results, tokenizer):
    """Output results to stderr as they are collected"""
    tokens = []
    last_len = 0

    for result in results:
        tokens.append(result.token_id)
        text = tokenizer.decode(tokens)
        sys.stderr.write(text[last_len:])
        sys.stderr.flush()
        last_len = len(text)

    sys.stderr.write("\n\n")
    sys.stderr.flush()
    return tokens


def generate(
    instructions: List[str],
    max_tokens: int = 200,
    temperature: float = 0.1,
    topk: int = 1,
    repetition_penalty: float = 0.0,
    prefix: str = "",
    suppress: List[str] = [],
    model: str = "instruct",
    stream: bool = False,
):
    """Generates completions for a prompt

    This may use a local model, or it may make an API call to an external
    model if API keys are available.

    >>> generate(["What is the capital of France?"])
    ['...Paris...']

    >>> list(generate(["What is the capital of France?"], stream=True))
    ['...Paris...']
    """
    tokenizer, model = get_model(model)

    start_time = perf_counter()

    suppress = [tokenizer.encode(s, add_special_tokens=False).tokens for s in suppress]

    model_info = get_model_info("instruct")

    fmt = model_info.get("prompt_fmt", "{instruction}")

    if repetition_penalty == 0.0:
        repetition_penalty = model_info.get("repetition_penalty", 1.3)

    prompts = [fmt.replace("{instruction}", inst) for inst in instructions]
    truncated_prompts = [truncate_prompt(p) for p in prompts]

    prompts_tok = [tokenizer.encode(p).tokens for p in truncated_prompts]

    outputs_ids = []
    if hasattr(model, "translate_batch"):
        prefix = tokenizer.encode(prefix, add_special_tokens=False).tokens
        if stream or (config["echo"] and len(prompts_tok) == 1):
            results = model.generate_tokens(
                prompts_tok[0],
                target_prefix=prefix,
                repetition_penalty=repetition_penalty,
                max_decoding_length=max_tokens,
                sampling_temperature=temperature,
                sampling_topk=topk,
                suppress_sequences=suppress,
            )

            if stream:
                return stream_results(results, tokenizer)
            else:
                outputs_ids = [echo_results(results, tokenizer)]
        else:
            results = model.translate_batch(
                prompts_tok,
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
        if stream or (config["echo"] and len(prompts_tok) == 1):
            results = model.generate_tokens(
                prompts_tok,
                repetition_penalty=repetition_penalty,
                max_length=max_tokens,
                sampling_temperature=temperature,
                sampling_topk=topk,
                suppress_sequences=suppress,
            )

            if stream:
                return stream_results(results, tokenizer)
            else:
                outputs_ids = [echo_results(results, tokenizer)]
        else:
            results = model.generate_batch(
                prompts_tok,
                repetition_penalty=repetition_penalty,
                max_length=max_tokens,
                sampling_temperature=temperature,
                sampling_topk=topk,
                suppress_sequences=suppress,
                beam_size=1,
                include_prompt_in_result=False,
            )
            outputs_ids = [r.sequences_ids[0] for r in results]

    model_info["requests"] = model_info.get("requests", 0) + len(prompts)

    in_toks = sum(len(p) for p in prompts_tok)
    model_info["input_tokens"] = model_info.get("input_tokens", 0) + in_toks

    out_toks = sum(len(o) for o in outputs_ids)
    model_info["output_tokens"] = model_info.get("output_tokens", 0) + out_toks

    elapsed_time = perf_counter() - start_time
    model_info["runtime"] = model_info.get("runtime", 0) + elapsed_time

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

    model_info = get_model_info("instruct")
    fmt = model_info.get("prompt_fmt", "{instruction}")
    inputs = [fmt.replace("{instruction}", inst) for inst in inputs]
    inputs = [truncate_prompt(i) for i in inputs]

    targ_tok = [tokenizer.encode(t, add_special_tokens=False).tokens for t in targets]
    targ_tok *= len(inputs)

    in_tok = []
    for input in inputs:
        toks = [tokenizer.encode(input).tokens]
        in_tok += toks * len(targets)

    if "Generator" in str(type(model)):
        scores = model.score_batch([i + t for i, t in zip(in_tok, targ_tok)])
    else:
        scores = model.score_batch(in_tok, target=targ_tok)

    ret = []
    for i in range(0, len(inputs) * len(targets), len(targets)):
        logprobs = [sum(r.log_probs) for r in scores[i : i + len(targets)]]
        results = sorted(zip(targets, logprobs), key=lambda r: -r[1])
        ret.append([r[0] for r in results])

    return ret
