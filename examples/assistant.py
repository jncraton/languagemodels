""" A simple assistant

The assistant uses information retrieval to obtain context from a small set
of stored documents. The included information is the current weather, current
date, and a brief summary of the Python programming language and the planet
Saturn.

A number of demonstration question are completed to demonstrate the available
functionality.
"""

import languagemodels as lm


def assist(question):
    context = lm.get_doc_context(question).replace(": ", " - ")

    return lm.do(f"Answer using context: {context} Question: {question}")


lat, lon = (41.8, -87.6)

lm.store_doc(lm.get_date())
lm.store_doc(lm.get_weather(lat, lon))
lm.store_doc(lm.get_wiki("Python language"))
lm.store_doc(lm.get_wiki("Planet Saturn"))

questions = [
    "What day of the week is it?",
    "Is it going to rain today?",
    "What time is it?",
    "Who created Python?",
    "How many moon does Saturn have?",
]

for question in questions:
    print(f"{question} {assist(question)}")
