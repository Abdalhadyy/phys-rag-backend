from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "nuclear-physics-rag"

def create_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index {INDEX_NAME} created ✅")
    else:
        print(f"Index {INDEX_NAME} already exists ✅")

def store_chunks(chunks: list, embeddings: list, pdf_name: str):
    index = pc.Index(INDEX_NAME)
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{pdf_name}_chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk, "source": pdf_name}
        })
    index.upsert(vectors=vectors)
    print(f"Stored {len(vectors)} chunks in Pinecone ✅")