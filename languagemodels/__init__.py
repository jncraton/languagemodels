import requests
import datetime
import json
import re
from typing import overload

from languagemodels.config import config
from languagemodels.preprocess import get_html_paragraphs
from languagemodels.inference import (
    generate,
    rank_instruct,
    list_tokens,
)
from languagemodels import embeddings

docs = embeddings.RetrievalContext()


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

    result = generate(
        ["Write a sentence"],
        prefix=prompt,
        max_tokens=config["max_tokens"],
        temperature=0.7,
        topk=40,
    )[0]

    if result.startswith(prompt):
        prefix_length = len(prompt)
        return result[prefix_length:]
    else:
        return result


@overload
def do(prompt: list) -> list:
    ...


@overload
def do(prompt: str) -> str:
    ...


def do(prompt, choices=None):
    """Follow a single-turn instructional prompt

    :param prompt: Instructional prompt(s) to follow
    :param choices: If provided, outputs are restricted to values in choices
    :return: Completion returned from the language model

    Note that this function is overloaded to return a list of results if
    a list if of prompts is provided and a single string if a single
    prompt is provided as a string

    Examples:

    >>> do("Translate Spanish to English: Hola mundo!") #doctest: +SKIP
    'Hello world!'

    >>> do("Pick the planet from the list: baseball, Texas, Saturn")
    '...Saturn...'

    >>> do(["Pick the planet from the list: baseball, Texas, Saturn"] * 2)
    ['...Saturn...', '...Saturn...']

    >>> do(["Say red", "Say blue"], choices=["red", "blue"])
    ['red', 'blue']

    >>> do("Classify as positive or negative: LLMs are bad",
    ... choices=["Positive", "Negative"])
    'Negative'

    >>> do("Classify as positive or negative: LLMs are great",
    ... choices=["Positive", "Negative"])
    'Positive'
    """

    prompts = [prompt] if isinstance(prompt, str) else prompt

    if choices:
        results = [r[0] for r in rank_instruct(prompts, choices)]
    else:
        results = generate(prompts, max_tokens=config["max_tokens"], topk=1)

    return results[0] if isinstance(prompt, str) else results


@overload
def embed(doc: list) -> list:
    ...


@overload
def embed(doc: str) -> str:
    ...


def embed(doc):
    """Create embedding for a document

    :param doc: Document(s) to embed
    :return: Embedding

    Note that this function is overloaded to return a list of embeddings if
    a list if of docs is provided and a single embedding if a single
    doc is provided as a string

    Examples:

    >>> embed("Hello, world")
    [-0.0...]

    >>> embed(["Hello", "world"])
    [[-0.0...]]
    """

    docs = [doc] if isinstance(doc, str) else doc

    # Create embeddings and convert to lists of floats
    emb = [[float(n) for n in e] for e in embeddings.embed(docs)]

    return emb[0] if isinstance(doc, str) else emb


def store_doc(doc: str, name: str = "") -> None:
    """Store document for later retrieval

    :param doc: A plain text document to store.
    :param name: Optional name for the document. This is used as a chunk prefix.

    Examples:

    >>> store_doc("The sky is blue.")
    """
    docs.store(doc, name)


def load_doc(query: str) -> str:
    """Load a matching document

    A single document that best matches `query` will be returned.

    :param query: Query to compare to stored documents
    :return: Content of the closest matching document

    Examples:

    >>> store_doc("Paris is in France.")
    >>> store_doc("The sky is blue.")
    >>> load_doc("Where is Paris?")
    'Paris is in France.'
    """
    return docs.get_match(query)


def get_doc_context(query: str) -> str:
    """Loads context from documents

    A string representing the most relevant content from all stored documents
    will be returned. This may be a blend of chunks from multiple documents.

    :param query: Query to compare to stored documents
    :return: Up to 128 tokens of context

    Examples:

    >>> store_doc("Paris is in France.")
    >>> store_doc("Paris is nice.")
    >>> store_doc("The sky is blue.")
    >>> get_doc_context("Where is Paris?")
    'Paris is in France.\\n\\nParis is nice.'
    """
    return docs.get_context(query)


def get_web(url: str) -> str:
    """
    Return the text of paragraphs from a web page

    :param url: The URL to load
    :return str: Plain text content from the URL

    Note that it is difficult to return only the human-readable
    content from an HTML page. This function takes a basic and quick
    approach. It will not work perfectly on all sites, but will
    often do a reasonable job of returning the plain text content
    of a page.

    If the `url` points to a plain text page, the page content
    will be returned verbatim.
    """

    res = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; languagemodels)"}
    )

    if "text/plain" in res.raw.getheader("content-type"):
        return res.text
    elif "text/html" in res.raw.getheader("content-type"):
        return get_html_paragraphs(res.text)

    return ""


def get_wiki(topic: str) -> str:
    """
    Return Wikipedia summary for a topic

    This function ignores the complexity of disambiguation pages and simply
    returns the first result that is not a disambiguation page

    :param topic: Topic to search for on Wikipedia
    :return: Text content of the lead section of the most popular matching article

    Examples:

    >>> get_wiki('Python language')
    'Python is a high-level...'

    >>> get_wiki('Chemistry')
    'Chemistry is the scientific study...'
    """

    url = "https://api.wikimedia.org/core/v1/wikipedia/en/search/title"
    response = requests.get(url, params={"q": topic, "limit": 5})
    response = json.loads(response.text)

    for page in response["pages"]:
        wiki_result = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts|pageprops&"
            f"exintro&redirects=1&titles={page['title']}&format=json"
        ).json()

        first = wiki_result["query"]["pages"].popitem()[1]
        if "disambiguation" in first["pageprops"]:
            continue

        summary = first["extract"]

        cutoffs = [
            "See_also",
            "Notes",
            "References",
            "Further_reading",
            "External_links",
        ]

        for cutoff in cutoffs:
            summary = summary.split(f'<span id="{cutoff}">', 1)[0]

        summary = re.sub(r"<p>", "\n\n", summary, flags=re.I)
        summary = re.sub(r"<!\-\-.*?\-\->", "", summary, flags=re.I | re.DOTALL)
        summary = re.sub(r"<.*?>", "", summary, flags=re.I)
        summary = re.sub(r"\s*[\n\r]+\s*[\r\n]+[\s\r\n]*", "\n\n", summary, flags=re.I)
        summary = summary.strip()
        return summary
    else:
        return "No matching wiki page found."


def get_weather(latitude, longitude):
    """Fetch the current weather for a supplied longitude and latitude

    Weather is provided by the US government and this function only supports
    locations in the United States.

    :param latitude: Latitude value representing this location
    :param longitude: Longitude value representing this location
    :return: Plain text description of the current weather forecast

    Examples:

    >>> get_weather(41.8, -87.6) # doctest: +SKIP
    'Scattered showers and thunderstorms before 1pm with a high of 73.'
    """

    res = requests.get(f"https://api.weather.gov/points/{latitude},{longitude}")
    points = json.loads(res.text)
    forecast_url = points["properties"]["forecast"]

    res = requests.get(forecast_url)
    forecast = json.loads(res.text)
    current = forecast["properties"]["periods"][0]

    return current["detailedForecast"]


def get_date() -> str:
    """Returns the current date and time in natural language

    >>> get_date() # doctest: +SKIP
    'Friday, May 12, 2023 at 09:27AM'
    """

    now = datetime.datetime.now()

    return now.strftime("%A, %B %d, %Y at %I:%M%p")


def print_tokens(prompt: str) -> None:
    """Prints a list of tokens in a prompt

    :param prompt: Prompt to use as input to tokenizer
    :return: Nothing

    Examples:

    >>> print_tokens("Hello world") # doctest: +SKIP
    ' Hello' (token 8774)
    ' world' (token 296)

    >>> print_tokens("Hola mundo") # doctest: +SKIP
    ' Hol' (token 5838)
    'a' (token 9)
    ' mun' (token 13844)
    'd' (token 26)
    'o' (token 32)
    """

    tokens = list_tokens(prompt)

    for token in tokens:
        print(f"'{token[0].replace('▁', ' ')}' (token {token[1]})")


def count_tokens(prompt: str) -> None:
    """Counts tokens in a prompt

    :param prompt: Prompt to use as input to tokenizer
    :return: Nothing

    Examples:

    >>> count_tokens("Hello world")
    2

    >>> count_tokens("Hola mundo") # doctest: +SKIP
    5
    """

    return len(list_tokens(prompt))


def set_max_ram(value):
    """Sets max allowed RAM

    This value takes priority over environment variables

    Returns the numeric value set in GB

    >>> set_max_ram(16)
    16.0

    >>> set_max_ram('512mb')
    0.5
    """

    config["max_ram"] = value

    return config["max_ram"]


def require_model_license(match_re):
    """Require models to match supplied regex

    This can be used to enforce certain licensing constraints when using this
    package.
    """
    config["model_license"] = match_re
