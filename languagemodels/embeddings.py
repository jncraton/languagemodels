from sentence_transformers import SentenceTransformer, util
import numpy

model = None


def encode(docs):
    """Encode a list of documents into their embeddings"""
    global model

    if not model:
        model = SentenceTransformer(
            "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6"
        )

    return model.encode(docs)


def get_dot_scores(query, docs):
    """Calculate similarity between query and a set of docs

    This is implemented by computing embeddings for the query and all
    documents in `docs`. Once embeddings are computed, the dot
    product is used to compute the similarity between each of the
    docs and the query. The text content of the most similar document is
    returned.
    """
    query_emb = encode(query)
    doc_emb = encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = list(zip(docs, scores))

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return doc_score_pairs


class RetrievalContext:
    """
    Provides a context for document retrieval

    Documents are embedded and cached for later search.

    Example usage:

    >>> rc = RetrievalContext()
    >>> rc.store("The sky is blue.")
    >>> rc.store("Paris is in France.")
    >>> rc.store("Mars is a planet.")
    >>> rc.get_match("Where is Paris?")
    'Paris is in France.'

    >>> rc.clear()
    >>> rc.get_match("Where is Paris?")
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.docs = []
        self.embeddings = None

    def store(self, doc):
        if doc not in self.docs:
            self.docs.append(doc)
            embedding = encode([doc])
            if isinstance(self.embeddings, numpy.ndarray):
                self.embeddings = numpy.concatenate((self.embeddings, embedding))
            else:
                self.embeddings = embedding

    def get_match(self, query):
        if len(self.docs) == 0:
            return None

        scores = util.dot_score(encode([query])[0], self.embeddings)[0].cpu().tolist()

        doc_score_pairs = list(zip(self.docs, scores))

        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[0][0]
