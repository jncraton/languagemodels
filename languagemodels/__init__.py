import requests
import json
import datetime

from languagemodels.inference import generate_instruct, get_pipeline
from languagemodels.embeddings import RetrievalContext

docs = RetrievalContext()


def do(prompt):
    """Follow a single-turn instructional prompt

    Examples:

    >>> do("Translate to Spanish: Hello, world!")
    'Hola, mundo!'

    >>> do("Pick the sport: baseball, texas, chemistry")
    'baseball'

    >>> do("Is the following positive or negative: I love Star Trek.")
    'positive'

    >>> do("Does this make sense: The course is jumping well.")
    'no'
    """
    return generate_instruct(prompt, max_tokens=200)


def complete(prompt):
    """Provide one completion for a given open-ended prompt

    Examples:

    >>> complete("Luke thought that he") #doctest: +SKIP
    'was going to be a doctor.'

    >>> complete("There are many mythical creatures who") #doctest: +SKIP
    'are able to fly'

    >>> complete("She hid in her room until") #doctest: +SKIP
    'she was sure she was safe'
    """
    return generate_instruct(prompt, max_tokens=200)


def chat(userprompt):
    """Respond to a prompt as a chat agent

    >>> chat("What is Mars?") #doctest: +SKIP
    'Mars is a planet in the solar system.'

    >>> chat("Who is Obama?") #doctest: +SKIP
    'Obama was president of the United States.'

    >>> chat("Where is Berlin?") #doctest: +SKIP
    'Berlin is located in Germany.'
    """

    now = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M%p")

    prompt = (
        f"Currently {now}.\n"
        f"Assistant responses are true helpful and harmless.\n"
        f"User: {userprompt}\n"
        f"Assistant: "
    )

    return do(prompt)


def store_doc(doc):
    """Store document for later retrieval

    >>> store_doc("The sky is blue.")
    """
    docs.store(doc)


def search_docs(query):
    """Search stored documents

    >>> store_doc("The sky is blue.")
    >>> store_doc("Paris is in France.")
    >>> search_docs("Where is Paris?")
    'Paris is in France.'
    """
    return docs.get_match(query)


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


def classify(doc, label1, label2):
    """Returns true of a supplied string is positive

    >>> classify("I love you!","positive","negative")
    'positive'
    >>> classify("That book was fine.","positive","negative")
    'positive'
    >>> classify("That movie was terrible.","positive","negative")
    'negative'
    """

    classifier = get_pipeline(
        "zero-shot-classification", "valhalla/distilbart-mnli-12-1"
    )

    result = classifier(doc, [label1, label2])

    top = max(zip(result["scores"], result["labels"]), key=lambda r: r[0])

    return top[1]
