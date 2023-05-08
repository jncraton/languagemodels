import requests
import os
from transformers import pipeline


class InferenceException(Exception):
    pass


modelcache = {}


def generate_instruct(prompt, max_tokens=200):
    if os.environ.get("textsynth-api-key"):
        response = requests.post(
            "https://api.textsynth.com/v1/engines/flan_t5_xxl/completions",
            headers={"Authorization": "Bearer " + os.environ.get("textsynth-api-key")},
            json={"prompt": prompt, "max_tokens": max_tokens},
        )
        resp = response.json()
        if "text" in resp:
            return resp["text"]
        else:
            raise InferenceException(f"TextSynth error: {resp}")

    generate = get_pipeline("text2text-generation", "google/flan-t5-large")

    return generate(prompt)[0]["generated_text"]


def get_pipeline(task, model):
    if model not in modelcache:
        modelcache[model] = pipeline(
            task, model=model, model_kwargs={"low_cpu_mem_usage": True}
        )

    return modelcache[model]
