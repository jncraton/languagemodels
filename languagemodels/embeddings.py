import numpy as np

from languagemodels.inference import get_model


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

    >>> rc.get_embedding("I love Python!")[-3:]
    array([0.116..., 0.128..., 0.023...], dtype=float32)
    """

    def __init__(self):
        self.clear()
        self.tokenizer, self.model = get_model("jncraton/all-MiniLM-L6-v2-ct2-int8")

    def clear(self):
        self.docs = []
        self.embeddings = []

    def get_embedding(self, doc):
        """Gets embeddings for a document"""
        tokens = self.tokenizer.encode(doc).ids
        output = self.model.forward_batch([tokens])
        embedding = np.mean(np.array(output.last_hidden_state), axis=1)[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def store(self, doc):
        if doc not in self.docs:
            embedding = self.get_embedding(doc)
            self.embeddings.append(embedding)
            self.docs.append(doc)

    def get_match(self, query):
        if len(self.docs) == 0:
            return None

        query_embedding = self.get_embedding(query)

        scores = [cosine_similarity(query_embedding, e) for e in self.embeddings]
        doc_score_pairs = list(zip(self.docs, scores))

        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[0][0]
