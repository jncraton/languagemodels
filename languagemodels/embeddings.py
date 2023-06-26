import numpy as np

from languagemodels.models import get_model


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


def embed(doc):
    """Gets embeddings for a document

    >>> embed("I love Python!")[-3:]
    array([0.1..., 0.1..., 0.0...], dtype=float32)
    """
    tokenizer, model = get_model("embedding")

    tokens = tokenizer.encode(doc).ids
    output = model.forward_batch([tokens[:512]])
    embedding = np.mean(np.array(output.last_hidden_state), axis=1)[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def search(query, docs):
    """Return docs sorted by match against query

    :param query: Input to match in search
    :param docs: List of docs to search against
    :return: List of (doc_num, score) tuples sorted by score descending
    """

    query_embedding = embed(query)

    scores = [cosine_similarity(query_embedding, d.embedding) for d in docs]

    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


class Document:
    """
    A document used for semantic search

    Documents have content and an embedding that is used to match the content
    against other semantically similar documents.
    """

    def __init__(self, content):
        self.content = content
        self.embedding = embed(content)


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

    >>> rc.clear()
    >>> rc.store(' '.join(['Python'] * 232))
    >>> len(rc.chunks)
    4

    >>> rc.get_context("What is Python?")
    'Python Python Python...'

    >>> [len(c.split()) for c in rc.chunks]
    [64, 64, 64, 64]

    >>> len(rc.get_context("What is Python?").split())
    128
    """

    def __init__(self, chunk_size=64, chunk_overlap=8):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clear()

    def clear(self):
        self.docs = []
        self.chunks = []

    def store(self, doc, name=""):
        """Stores a document along with embeddings

        This stores both the document as well as document chunks

        >>> rc = RetrievalContext()
        >>> rc.clear()
        >>> rc.store(' '.join(['Python'] * 233))
        >>> len(rc.chunks)
        5

        >>> rc.clear()
        >>> rc.store(' '.join(['Python'] * 232))
        >>> len(rc.chunks)
        4

        >>> rc.clear()
        >>> rc.store('Python')
        >>> len(rc.chunks)
        1

        >>> rc.clear()
        >>> rc.store('It is a language.', 'Python')
        >>> len(rc.chunks)
        1
        >>> [c.content for c in rc.chunks]
        ['Python: It is a language.']

        >>> rc = RetrievalContext()
        >>> rc.clear()
        >>> rc.store(' '.join(['details'] * 225), 'Python')
        >>> len(rc.chunks)
        5

        >>> rc.clear()
        >>> rc.store(' '.join(['details'] * 224), 'Python')
        >>> len(rc.chunks)
        4
        >>> [c.content for c in rc.chunks]
        ['Python: details details details...']
        """

        if doc not in self.docs:
            self.docs.append(Document(doc))
            self.store_chunks(doc, name)

    def store_chunks(self, doc, name=""):
        # Note that the tokenzier used here is from the generative model
        # This is used for token counting for the context, not for tokenization
        # before embedding
        generative_tokenizer, _ = get_model("instruct", tokenizer_only=True)

        tokens = generative_tokenizer.encode(doc, add_special_tokens=False).ids

        name_tokens = []

        if name:
            name_tokens = generative_tokenizer.encode(
                f"{name}:", add_special_tokens=False
            ).ids

        i = 0
        chunk = name_tokens.copy()
        while i < len(tokens):
            token = tokens[i]
            chunk.append(token)
            i += 1

            # Begin looking for probable sentence when half of target size

            full = len(chunk) == self.chunk_size
            half_full = len(chunk) > self.chunk_size / 2
            eof = i == len(tokens)
            sep = token in [".", "!", "?", ")."]

            if eof or full or (half_full and sep):
                # Store tokens and start next chunk
                text = generative_tokenizer.decode(chunk)
                self.chunks.append(Document(text))
                chunk = name_tokens.copy()
                if full and not eof:
                    # If the heuristic didn't get a semantic boundary, overlap
                    # next chunk to provide some context
                    i -= self.chunk_overlap
                    i = max(0, i)

    def get_context(self, query, max_tokens=128):
        """Gets context matching a query

        Context is capped by token length and is retrieved from stored
        document chunks
        """

        if len(self.chunks) == 0:
            return None

        results = search(query, self.chunks)

        chunks = []
        tokens = 0

        generative_tokenizer, _ = get_model("instruct", tokenizer_only=True)

        for chunk_id, score in results:
            chunk = self.chunks[chunk_id].content
            chunk_tokens = len(generative_tokenizer.encode(chunk, add_special_tokens=False).tokens)
            if tokens + chunk_tokens <= max_tokens and score > 0.1:
                chunks.append(chunk)
                tokens += chunk_tokens

        context = "\n\n".join(chunks)

        return context

    def get_match(self, query):
        if len(self.docs) == 0:
            return None

        return self.docs[search(query, self.docs)[0][0]].content
