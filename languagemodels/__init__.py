import requests
import datetime
import json

from languagemodels.inference import generate_instruct, get_pipeline
from languagemodels.embeddings import RetrievalContext

docs = RetrievalContext()


def complete(prompt: str) -> str:
    """Provide one completion for a given open-ended prompt

    :param prompt: Prompt to use as input to the model
    :return: Completion returned from the language model

    Examples:

    >>> complete("Luke thought that he") #doctest: +SKIP
    'was going to be a doctor.'

    >>> complete("There are many mythical creatures who") #doctest: +SKIP
    'are able to fly'

    >>> complete("She hid in her room until") #doctest: +SKIP
    'she was sure she was safe'
    """
    return generate_instruct(prompt, max_tokens=200, temperature=0.7)


def do(prompt: str) -> str:
    """Follow a single-turn instructional prompt

    :param prompt: Instructional prompt to follow
    :return: Completion returned from the language model

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


def extract_answer(question: str, context: str) -> str:
    """Extract an answer to a `question` from a provided `context`

    The returned answer will always be a substring extracted from `context`.
    It may not always be a correct or meaningful answer, but it will never be
    an arbitrary hallucination.

    :param question: A question to answer using knowledge from context
    :param context: Knowledge used to answer the question
    :return: Answer to the question.

    >>> extract_answer("What color is the ball?", "There is a green ball and a red box")
    'green'
    >>> extract_answer("Who created Python?", fetch_wiki('Python'))
    'Guido van Rossum'
    """

    qa = get_pipeline("question-answering", "distilbert-base-cased-distilled-squad")

    return qa(question, context)["answer"]


def classify(doc: str, label1: str, label2: str) -> str:
    """Performs binary classification on an input

    :param doc: A plain text input document to classify
    :param label1: The first label to classify against
    :param label2: The second label to classify against
    :return: The closest matching class. The return value will always be
    `label1` or `label2`

    >>> classify("I love you!","positive","negative")
    'positive'
    >>> classify("That book was fine.","positive","negative")
    'positive'
    >>> classify("That movie was terrible.","positive","negative")
    'negative'
    >>> classify("The submarine is diving", "ocean", "space")
    'ocean'
    """

    classifier = get_pipeline(
        "zero-shot-classification", "valhalla/distilbart-mnli-12-1"
    )

    result = classifier(doc, [label1, label2])

    top = max(zip(result["scores"], result["labels"]), key=lambda r: r[0])

    return top[1]


def store_doc(doc: str) -> None:
    """Store document for later retrieval

    :param doc: A plain text document to store.

    >>> store_doc("The sky is blue.")
    """
    docs.store(doc)


def search_docs(query: str) -> str:
    """Search stored documents

    A single document that best matches `query` will be returned.

    :param prompt: Query to compare to stored documents
    :return: Content of the closest matching document

    >>> store_doc("The sky is blue.")
    >>> store_doc("Paris is in France.")
    >>> search_docs("Where is Paris?")
    'Paris is in France.'
    """
    return docs.get_match(query)


def fetch_wiki(topic: str) -> str:
    """
    Return Wikipedia summary for a topic

    This function ignores the complexity of disambiguation pages and simply
    returns the first result that is not a disambiguation page

    :param topic: Topic to search for on Wikipedia
    :return: Text content of the lead section of the most popular matching article

    >>> fetch_wiki('Python') # doctest: +ELLIPSIS
    'Python is a high-level...

    >>> fetch_wiki('Chemistry') # doctest: +ELLIPSIS
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
    else:
        return "No matching wiki page found."


def get_date() -> str:
    """Returns the current date and time in natural language

    >>> get_date() # doctest: +SKIP
    'Friday, May 12, 2023 at 09:27AM'
    """

    now = datetime.datetime.now()

    return now.strftime("%A, %B %d, %Y at %I:%M%p")
