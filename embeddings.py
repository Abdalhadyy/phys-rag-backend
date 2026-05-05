from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> list:
    embedding = model.encode(text)
    return embedding.tolist()

def get_embeddings_batch(chunks: list) -> list:
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        print(f"Chunk {i+1}/{len(chunks)} embedded ✅")
    return embeddings