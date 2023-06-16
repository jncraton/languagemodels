import numpy as np

from languagemodels.models import get_model


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
    array([0.1..., 0.1..., 0.0...], dtype=float32)

    >>> rc.clear()
    >>> rc.store('Python ' * 232)
    >>> len(rc.chunks)
    4

    >>> rc.get_context("What is Python?")
    'Python Python Python...'

    >>> len(rc.get_context("What is Python?").split())
    128
    """

    def __init__(self, chunk_size=64, chunk_overlap=8):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clear()

    def clear(self):
        self.docs = []
        self.embeddings = []
        self.chunks = []
        self.chunk_embeddings = []

    def get_embedding(self, doc):
        """Gets embeddings for a document"""
        tokenizer, model = get_model("embedding")

        tokens = tokenizer.encode(doc).ids
        output = model.forward_batch([tokens])
        embedding = np.mean(np.array(output.last_hidden_state), axis=1)[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def store(self, doc):
        """Stores a document along with embeddings

        This stores both the document as well as document chunks

        >>> rc = RetrievalContext()
        >>> rc.clear()
        >>> rc.store('Python ' * 233)
        >>> len(rc.chunks)
        5

        >>> rc.clear()
        >>> rc.store('Python ' * 232)
        >>> len(rc.chunks)
        4

        >>> rc.clear()
        >>> rc.store('Python')
        >>> len(rc.chunks)
        1
        """

        if doc not in self.docs:
            embedding = self.get_embedding(doc)
            self.embeddings.append(embedding)
            self.docs.append(doc)
            self.store_chunks(doc)

    def store_chunks(self, doc):
        # Note that the tokenzier used here is from the generative model
        # This is used for token counting for the context, not for tokenization
        # before embedding
        generative_tokenizer, _ = get_model("instruct")

        tokens = generative_tokenizer.EncodeAsPieces(doc)

        end = max(len(tokens) - self.chunk_overlap, 1)
        stride = self.chunk_size - self.chunk_overlap

        for i in range(0, end, stride):
            chunk = tokens[i : i + self.chunk_size]
            text = generative_tokenizer.Decode(chunk)
            embedding = self.get_embedding(text)
            self.chunk_embeddings.append(embedding)
            self.chunks.append(text)

    def get_context(self, query, max_tokens=128):
        """Gets context matching a query

        Context is capped by token length and is retrieved from stored
        document chunks
        """

        if len(self.chunks) == 0:
            return None

        query_embedding = self.get_embedding(query)

        scores = [cosine_similarity(query_embedding, e) for e in self.chunk_embeddings]
        doc_score_pairs = list(zip(self.chunks, scores))

        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        chunks = []
        tokens = 0

        generative_tokenizer, _ = get_model("instruct")

        for chunk, score in doc_score_pairs:
            chunk_tokens = len(generative_tokenizer.EncodeAsPieces(chunk))
            if tokens + chunk_tokens <= max_tokens and score > 0.1:
                chunks.append(chunk)
                tokens += chunk_tokens

        context = "\n\n".join(chunks)

        return context

    def get_match(self, query):
        if len(self.docs) == 0:
            return None

        query_embedding = self.get_embedding(query)

        scores = [cosine_similarity(query_embedding, e) for e in self.embeddings]
        doc_score_pairs = list(zip(self.docs, scores))

        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[0][0]
