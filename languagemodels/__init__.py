import requests
import datetime
import json

from languagemodels.inference import generate_instruct, get_pipeline, convert_chat
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


def chat(prompt: str) -> str:
    """Get new message from chat-optimized language model

    The `prompt` for this model is provided as a series of messages as a single
    plain-text string. Several special tokens are used to delineate chat
    messages.

    - `<|system|>` - Indicates the start of a system message providing
    instructions about how the assistant should behave.
    - `<|prompter|>` - Indicates the start of a prompter (typically user)
    message.
    - `<|assistant|>` - Indicates the start of an assistant message.
    - `<|endoftext|>` - Used to terminal all message types.

    A complete prompt may look something like this:

    ```
    <|system|>Assistant is helpful and harmless<|endoftext|>
    <|prompter|>What is the capital of Germany?<|endoftext|>
    <|assistant|>The capital of Germany is Berlin.<|endoftext|>
    <|prompter|>How many people live there?<|endoftext|>
    <|assistant|>
    ```

    The completion from the language model is returned.

    :param message: List of message as (role, content) tuples
    :return: Completion returned from the language model

    >>> chat("<|system|>It is 5:15pm. Assistant is helpful<|endoftext|>" \\
    ...      "<|prompter|>Do you know what time it is?<|endoftext|>" \\
    ...      "<|assistant|>")
    'It is 5:15pm.'
    """

    prompt = convert_chat(prompt)

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


def load_doc(query: str) -> str:
    """Load a matching document

    A single document that best matches `query` will be returned.

    :param query: Query to compare to stored documents
    :return: Content of the closest matching document

    >>> store_doc("The sky is blue.")
    >>> store_doc("Paris is in France.")
    >>> load_doc("Where is Paris?")
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


def fetch_weather(latitude, longitude):
    """Fetch the current weather for a supplied longitude and latitude

    Weather is provided by the US government and this function only supports
    locations in the United States.

    :param latitude: Latitude value representing this location
    :param longitude: Longitude value representing this location
    :return: Plain text description of the current weather forecast

    >>> fetch_weather(41.8, -87.6) # doctest: +SKIP
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
