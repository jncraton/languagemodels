from sentence_transformers import SentenceTransformer, util

model = None

def encode(docs):
    global model
    
    if not model:
        model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    return model.encode(docs)