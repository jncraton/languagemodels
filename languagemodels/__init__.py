import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import json

from languagemodels.inference import modelcache, get_model, get_tokenizer, generate_instruct, get_pipeline


def do(prompt):
    """Follow a single-turn instructional prompt

    Examples:

    >>> do("Translate to Spanish: Hello, world!")
    'Hola, mundo!'

    >>> do("Pick the sport: baseball, texas, chemistry")
    'baseball'

    >>> do("Is the following positive or negative: I love Star Trek.")
    'positive'
    """
    return generate_instruct(prompt, max_tokens=200)

def chat(userprompt):
    prompt = (
        f"System: Agent responses will be truthful, helpful, and harmless.\n"
        f"User: {userprompt}\n"
        f"Agent: "
    )

    return do(prompt)


def match(query, docs):
    """Return closest matching document in `docs` using semantic search

    >>> match("What is Mars?", ["Mars is a planet", "The sun is hot"])
    'Mars is a planet'

    >>> match("Where is Paris?", ["Paris is rainy", "Paris is in France"])
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


def search(topic):
    """
    Return Wikipedia summary for a topic

    This function ignores the complexity of disambiguation pages and simply
    returns the first result that is not a disambiguation page

    >>> search('Python') # doctest: +ELLIPSIS
    'Python is a high-level...

    >>> search('Chemistry') # doctest: +ELLIPSIS
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


def extract_answer(question, context):
    """Extract an answer to a question from a provided context

    >>> extract_answer("What color is the ball?", "There is a green ball and a red box")
    'green'
    >>> extract_answer("Who created Python?", search('Python'))
    'Guido van Rossum'
    """

    qa = get_pipeline("question-answering", "distilbert-base-cased-distilled-squad")

    return qa(question, context)["answer"]


def is_positive(doc):
    """Returns true of a supplied string is positive

    >>> is_positive("I love you!")
    True
    >>> is_positive("That book was fine.")
    True
    >>> is_positive("That movie was terrible.")
    False
    """

    model = "distilbert-base-uncased-finetuned-sst-2-english"

    if model not in modelcache:
        modelcache[model] = pipeline("text-classification", model=model)

    prediction = modelcache[model](doc)

    return prediction[0]["label"] == "POSITIVE"
