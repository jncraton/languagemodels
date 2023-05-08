import os
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import json


class InferenceException(Exception):
    pass


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


def generate_instruct(prompt):
    if os.environ.get("textsynth-api-key"):
        response = requests.post(
            "https://api.textsynth.com/v1/engines/flan_t5_xxl/completions",
            headers={"Authorization": "Bearer " + os.environ.get("textsynth-api-key")},
            json={"prompt": prompt, "max_tokens": 200},
        )
        resp = response.json()
        if "text" in resp:
            return resp["text"]
        else:
            raise InferenceException(f"TextSynth error: {resp}")

    model = get_model("google/flan-t5-large")
    tokenizer = get_tokenizer("google/flan-t5-large")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.2)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def chat(userprompt):
    prompt = (
        f"System: "
        f"Agent responses will be truthful, helpful, and harmless.\n"
        f"User: {userprompt}\n"
        f"Agent: "
    )

    return generate_instruct(prompt)


def search(query, docs):
    """
    >>> search("What is Mars?", ["Mars is a planet", "The sun is hot"])
    'Mars is a planet'

    >>> search("Where is Paris?", ["Paris is rainy", "Paris is in France"])
    'Paris is in France'
    """

    model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

    if model not in modelcache:
        modelcache[model] = SentenceTransformer(model)

    query_emb = modelcache[model].encode(query)
    doc_emb = modelcache[model].encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = list(zip(docs, scores))

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    return doc_score_pairs[0][0]


def get_wiki(topic):
    """
    Return Wikipedia summary for a topic

    This function ignores the complexity of disambiguation pages and simply
    returns the first result that is not a disambiguation page

    >>> get_wiki('Python') # doctest: +ELLIPSIS
    'Python is a high-level...

    >>> get_wiki('Chemistry') # doctest: +ELLIPSIS
    'Chemistry is the scientific study...
    """

    url = "https://api.wikimedia.org/core/v1/wikipedia/en/search/title"
    response = requests.get(url, params={"q": topic, "limit": 5})
    response = json.loads(response.text)

    for page in response["pages"]:
        wiki_result = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts|pageprops&"
            f"exintro&explaintext&redirects=1&titles={page['title']}&format=json"
        ).json()

        first = wiki_result["query"]["pages"].popitem()[1]
        if "disambiguation" in first["pageprops"]:
            continue

        summary = first["extract"]
        return summary
