from sentence_transformers import SentenceTransformer, util

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
