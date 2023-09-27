import numpy as np

from languagemodels.models import get_model, get_model_info


def embed(docs):
    """Get embeddings for a batch of documents

    >>> embed(["I love Python!"])[0].shape
    (384,)

    >>> embed(["I love Python!"])[0][-3:]
    array([0.1..., 0.1..., 0.0...], dtype=float32)
    """

    tokenizer, model = get_model("embedding")

    tokens = [tokenizer.encode(doc).ids[:512] for doc in docs]
    outputs = model.forward_batch(tokens)

    def mean_pool(last_hidden_state):
        embedding = np.mean(last_hidden_state, axis=0)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    return [mean_pool(lhs) for lhs in np.array(outputs.last_hidden_state)]


def search(query, docs):
    """Return docs sorted by match against query

    :param query: Input to match in search
    :param docs: List of docs to search against
    :return: List of (doc_num, score) tuples sorted by score descending
    """

    prefix = get_model_info("embedding")["query_prefix"]

    query_embedding = embed([f"{prefix}{query}"])[0]

    scores = [np.dot(query_embedding, d.embedding) for d in docs]

    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


def get_token_ids(doc):
    """Return list of token ids for a document

    Note that the tokenzier used here is from the generative model.

    This is used for token counting for the context, not for tokenization
    before embedding.
    """

    generative_tokenizer, _ = get_model("instruct", tokenizer_only=True)

    # We need to disable and re-enable truncation here
    # This allows us to tokenize very large documents
    # We won't be feeding the tokens themselves to a model, so this
    # shouldn't cause any problems.
    trunk = generative_tokenizer.truncation
    if trunk:
        generative_tokenizer.no_truncation()
    ids = generative_tokenizer.encode(doc, add_special_tokens=False).ids
    if trunk:
        generative_tokenizer.enable_truncation(
            trunk["max_length"], stride=trunk["stride"], strategy=trunk["strategy"]
        )

    return ids


def chunk_doc(doc, name="", chunk_size=64, chunk_overlap=8):
    """Break a document into chunks

    :param doc: Document to chunk
    :param name: Optional document name
    :param chunk_size: Length of individual chunks in tokens
    :param chunk_overlap: Number of tokens to overlap when breaking chunks
    :return: List of strings representing the chunks

    >>> chunk_doc("")
    []

    >>> chunk_doc("Hello")
    ['Hello']

    >>> chunk_doc("Hello " * 65)
    ['Hello Hello...', 'Hello...']

    >>> chunk_doc("Hello world. " * 24)[0]
    'Hello world. ...Hello world.'

    >>> len(chunk_doc("Hello world. " * 20))
    1

    >>> len(chunk_doc("Hello world. " * 24))
    2

    # Check to make sure sentences aren't broken on decimal points
    >>> chunk_doc(('z. ' + ' 37.468 ' * 5) * 3)[0]
    'z. 37.468 ...z.'
    """
    generative_tokenizer, _ = get_model("instruct", tokenizer_only=True)

    tokens = get_token_ids(doc)
    separators = [get_token_ids(t)[-1] for t in [".", "!", "?", ")."]]

    name_tokens = []

    label = f"From {name} document:" if name else ""

    if name:
        name_tokens = get_token_ids(label)

    i = 0
    chunks = []
    chunk = name_tokens.copy()
    while i < len(tokens):
        token = tokens[i]
        chunk.append(token)
        i += 1

        # Save the last chunk if we're done
        if i == len(tokens):
            chunks.append(generative_tokenizer.decode(chunk))
            break

        if len(chunk) == chunk_size:
            # Backtrack to find a reasonable cut point
            for j in range(1, chunk_size - chunk_overlap * 2):
                if chunk[chunk_size - j] in separators:
                    ctx = generative_tokenizer.decode(
                        chunk[chunk_size - j : chunk_size - j + 2]
                    )
                    if " " in ctx:
                        # Found a good separator
                        text = generative_tokenizer.decode(chunk[: chunk_size - j + 1])
                        chunks.append(text)
                        chunk = name_tokens + chunk[chunk_size - j + 1 :]
                        break
            else:
                # No semantically meaningful cutpoint found
                # Default to a hard cut
                text = generative_tokenizer.decode(chunk)
                chunks.append(text)
                # Share some overlap with next chunk
                overlap = max(
                    chunk_overlap, chunk_size - len(name_tokens) - (len(tokens) - i)
                )
                chunk = name_tokens + chunk[-overlap:]

    return chunks


class Document:
    """
    A document used for semantic search

    Documents have content and an embedding that is used to match the content
    against other semantically similar documents.
    """

    def __init__(self, content, name="", embedding=None):
        self.content = content
        self.embedding = embedding if embedding is not None else embed([content])[0]
        self.name = name


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
    >>> rc.store(' '.join(['Python'] * 4096))
    >>> len(rc.chunks)
    73

    >>> rc.clear()
    >>> rc.store(' '.join(['Python'] * 232))
    >>> len(rc.chunks)
    4

    >>> rc.get_context("What is Python?")
    'Python Python Python...'

    >>> [len(c.content.split()) for c in rc.chunks]
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
        ['From Python document: It is a language.']

        >>> rc = RetrievalContext()
        >>> rc.clear()
        >>> rc.store(' '.join(['details'] * 217), 'Python')
        >>> len(rc.chunks)
        5

        >>> rc.clear()
        >>> rc.store(' '.join(['details'] * 216), 'Python')
        >>> len(rc.chunks)
        4
        >>> [c.content for c in rc.chunks]
        ['From Python document: details details details...']
        """

        if doc not in self.docs:
            self.docs.append(Document(doc))
            self.store_chunks(doc, name)

    def store_chunks(self, doc, name=""):
        chunks = chunk_doc(doc, name, self.chunk_size, self.chunk_overlap)

        embeddings = embed(chunks)

        for embedding, chunk in zip(embeddings, chunks):
            self.chunks.append(Document(chunk, embedding=embedding))

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

        for chunk_id, score in results:
            chunk = self.chunks[chunk_id].content
            chunk_tokens = len(get_token_ids(chunk))
            if tokens + chunk_tokens <= max_tokens and score > 0.1:
                chunks.append(chunk)
                tokens += chunk_tokens

        context = "\n\n".join(chunks)

        return context

    def get_match(self, query):
        if len(self.docs) == 0:
            return None

        return self.docs[search(query, self.docs)[0][0]].content
