from sentence_transformers import CrossEncoder
import os

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, chunks: list) -> list:
    # جهّز الأزواج
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # احسب scores
    scores = reranker_model.predict(pairs)
    
    # أضف الـ scores للـ chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    
    # رتّب من الأعلى للأقل
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    print(f"Re-ranking {len(chunks)} chunks ✅")
    return reranked