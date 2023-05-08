from sentence_transformers import SentenceTransformer, util

model = None

def encode(docs):
    """ Encode a list of documents into their embeddings
    """
    global model
    
    if not model:
        model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    return model.encode(docs)

def get_dot_scores(query, docs):
    """ Calculate dot products between a query a set of docs in embedding space
    """
    query_emb = encode(query)
    doc_emb = encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = list(zip(docs, scores))

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return doc_score_pairs
