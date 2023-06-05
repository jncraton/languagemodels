import tempfile
import re

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser, OrGroup


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
        schema = Schema(content=TEXT(stored=True))
        self.index = create_in(tempfile.gettempdir(), schema)

    def store(self, doc):
        writer = self.index.writer()
        writer.add_document(content=doc)
        writer.commit()

    def get_match(self, query):
        with self.index.searcher() as searcher:
            query = re.sub(r'[,\.\?\!\:\;]', ' ', query)
            qp = QueryParser("content", self.index.schema, group=OrGroup)
            query = qp.parse(query)
            results = searcher.search(query)

            try:
                return results[0]["content"]
            except IndexError:
                return None
