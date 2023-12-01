import requests
import datetime
import json
import re

from languagemodels.config import config
from languagemodels.inference import (
    generate_instruct,
    generate_code,
    rank_instruct,
    parse_chat,
    list_tokens,
)
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

    result = generate_instruct(
        "Write a sentence", prefix=prompt,
        max_tokens=config["max_tokens"], temperature=0.7, topk=40
    )

    if result.startswith(prompt):
        prefix_length = len(prompt)
        return result[prefix_length:]
    else:
        return result


def do(prompt: str) -> str:
    """Follow a single-turn instructional prompt

    :param prompt: Instructional prompt to follow
    :return: Completion returned from the language model

    Examples:

    >>> do("Translate Spanish to English: Hola mundo!") #doctest: +SKIP
    'Hello world!'

    >>> do("Pick the sport from the list: baseball, texas, chemistry")
    'Baseball.'

    >>> do("Is the following positive or negative: I love Star Trek.")
    'Positive.'
    """
    result = generate_instruct(prompt, max_tokens=config["max_tokens"], topk=1)

    if len(result.split()) == 1:
        result = result.title()

        if result[-1] not in (".", "!", "?"):
            result = result + "."

    return result


def chat(prompt: str) -> str:
    """Get new message from chat-optimized language model

    The `prompt` for this model is provided as a series of messages as a single
    plain-text string. Several special tokens are used to delineate chat
    messages.

    - `system:` - Indicates the start of a system message providing
    instructions about how the assistant should behave.
    - `user:` - Indicates the start of a prompter (typically user)
    message.
    - `assistant:` - Indicates the start of an assistant message.

    A complete prompt may look something like this:

    ```
    Assistant is helpful and harmless

    User: What is the capital of Germany?

    Assistant: The capital of Germany is Berlin.

    User: How many people live there?

    Assistant:
    ```

    The completion from the language model is returned.

    :param message: Prompt using formatting described above
    :return: Completion returned from the language model

    Examples:

    >>> chat('''
    ...      System: Respond as a helpful assistant. It is 5:00pm.
    ...
    ...      User: What time is it?
    ...
    ...      Assistant:
    ...      ''')
    '...5:00pm...'
    """

    messages = parse_chat(prompt)

    # Suppress starts of all assistant messages to avoid repeat generation
    suppress = [
        "Assistant: " + m["content"].split(" ")[0]
        for m in messages
        if m["role"] in ["assistant", "user"]
    ]

    # Suppress all user messages to avoid repeating them
    suppress += [m["content"] for m in messages if m["role"] == "user"]

    system_msgs = [m for m in messages if m["role"] == "system"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    user_msgs = [m for m in messages if m["role"] == "user"]

    # The current model is tuned on instructions and tends to get
    # lost if it sees too many questions
    # Use only the most recent user and assistant message for context
    # Keep all system messages
    messages = system_msgs + assistant_msgs[-1:] + user_msgs[-1:]

    rolemap = {
        "system": "System",
        "user": "Question",
        "assistant": "Assistant",
    }

    messages = [f"{rolemap[m['role']]}: {m['content']}" for m in messages]

    prompt = "\n\n".join(messages) + "\n\n" + "Assistant:"

    if prompt.startswith("System:"):
        prompt = prompt[7:].strip()

    response = generate_instruct(
        prompt,
        max_tokens=config["max_tokens"],
        repetition_penalty=1.3,
        temperature=0.3,
        topk=40,
        prefix="Assistant:",
        suppress=suppress,
    )

    # Remove duplicate assistant being generated
    if response.startswith("Assistant:"):
        response = response[10:]

    return response.strip()


def code(prompt: str) -> str:
    """Complete a code prompt

    This assumes that users are expecting Python completions. Default models
    are fine-tuned on Python where applicable.

    :param prompt: Code context to complete
    :return: Completion returned from the language model

    Examples:

    >>> code("# Print Hello, world!\\n")
    'print("Hello, world!")\\n'

    >>> code("def return_4():")
    '...return 4...'
    """
    result = generate_code(prompt, max_tokens=config["max_tokens"], topk=1)

    return result


def extract_answer(question: str, context: str) -> str:
    """Extract an answer to a `question` from a provided `context`

    The returned answer will always be a substring extracted from `context`.
    It may not always be a correct or meaningful answer, but it will never be
    an arbitrary hallucination.

    :param question: A question to answer using knowledge from context
    :param context: Knowledge used to answer the question
    :return: Answer to the question.

    Examples:

    >>> context = "There is a green ball and a red box"
    >>> extract_answer("What color is the ball?", context).lower()
    '...green...'

    >>> extract_answer("Who created Python?", get_wiki('Python')) #doctest: +SKIP
    '...Guido van Rossum...'
    """

    return generate_instruct(f"{context}\n\n{question}")


def classify(doc: str, label1: str, label2: str) -> str:
    """Performs binary classification on an input

    :param doc: A plain text input document to classify
    :param label1: The first label to classify against
    :param label2: The second label to classify against
    :return: The closest matching class. The return value will always be
    `label1` or `label2`

    Examples:

    >>> classify("I love you!","positive","negative")
    'positive'
    >>> classify("That book was good.","positive","negative")
    'positive'
    >>> classify("That movie was terrible.","positive","negative")
    'negative'
    >>> classify("The submarine is diving", "ocean", "land")
    'ocean'
    """

    results = rank_instruct(
        f"Classify as {label1} or {label2}: {doc}\n\nClassification:", [label1, label2]
    )

    return results[0]


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


def get_wiki(topic: str) -> str:
    """
    Return Wikipedia summary for a topic

    This function ignores the complexity of disambiguation pages and simply
    returns the first result that is not a disambiguation page

    :param topic: Topic to search for on Wikipedia
    :return: Text content of the lead section of the most popular matching article

    Examples:

    >>> get_wiki('Python')
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

    >>> print_tokens("Hello world")
    ' Hello' (token 8774)
    ' world' (token 296)

    >>> print_tokens("Hola mundo")
    ' Hol' (token 5838)
    'a' (token 9)
    ' mun' (token 13844)
    'd' (token 26)
    'o' (token 32)
    """

    tokens = list_tokens(prompt)

    for token in tokens:
        print(f"'{token[0].replace('▁',' ')}' (token {token[1]})")


def count_tokens(prompt: str) -> None:
    """Counts tokens in a prompt

    :param prompt: Prompt to use as input to tokenizer
    :return: Nothing

    Examples:

    >>> count_tokens("Hello world")
    2

    >>> count_tokens("Hola mundo")
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
