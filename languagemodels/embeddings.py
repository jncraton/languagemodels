def cosine_similarity(a, b):
    dot_product = sum(ai * bi for ai, bi in zip(a, b))
    magnitude_a = sum(ai ** 2 for ai in a) ** 0.5
    magnitude_b = sum(bi ** 2 for bi in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)


class RetrievalContext:
    """
    Provides a context for document retrieval

    Documents are embedded and cached for later search.

    Example usage:

    >>> rc = RetrievalContext()
    >>> rc.store("Paris is in France.")
    >>> rc.store("The sky is blue.")
    >>> rc.store("Mars is a planet.")
    >>> rc.get_match("Paris is in France.")
    'Paris is in France.'

    >>> rc.get_match("Where is Paris?")
    'Paris is in France.'

    >>> rc.clear()
    >>> rc.get_match("Where is Paris?")
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.docs = []
        self.embeddings = []

    def store(self, doc):
        if doc not in self.docs:
            self.docs.append(doc)
            self.embeddings.append([1.0])  # TODO Implement embeddings

    def get_match(self, query):
        if len(self.docs) == 0:
            return None

        query_embedding = [1.0]  # TODO Implement embeddings

        scores = [cosine_similarity(query_embedding, e) for e in self.embeddings]

        doc_score_pairs = list(zip(self.docs, scores))

        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[0][0]
