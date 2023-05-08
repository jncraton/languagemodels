model = None
cache = {}

def encode(docs):
    if not model:
        model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    return model.encode(docs)