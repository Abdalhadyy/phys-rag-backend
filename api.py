from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()  # ← هنا أولاً

from pdf_processor import extract_text_from_pdf, chunk_text
from embeddings import get_embeddings_batch
from pinecone_store import create_index, store_chunks
from retriever import retrieve
from context_builder import build_context
from query_rewriter import rewrite_query
from reranker import rerank
from multi_hop import multi_hop_retrieve
from context_compressor import compress_context
from llm import ask_llm, choose_model
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

app = FastAPI(title="Nuclear Physics RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

create_index()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    model_used: str

@app.get("/")
def root():
    return {"status": "Nuclear Physics RAG API is running ✅"}

@app.get("/papers")
async def get_papers():
    index = pc.Index("nuclear-physics-rag")
    results = index.query(
        vector=[0.0] * 384,
        top_k=100,
        include_metadata=True
    )
    papers = set()
    for match in results.matches:
        if 'source' in match.metadata:
            papers.add(match.metadata['source'])
    return {
        "papers": list(papers),
        "total": len(papers)
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(text)
        embeddings = get_embeddings_batch(chunks)
        store_chunks(chunks, embeddings, file.filename)
        return {
            "status": "success ✅",
            "filename": file.filename,
            "chunks": len(chunks)
        }
    finally:
        os.unlink(tmp_path)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query_text = request.query
    rewritten = rewrite_query(query_text)
    chunks = multi_hop_retrieve(rewritten, hops=2)
    reranked = rerank(rewritten, chunks)
    context = build_context(reranked[:4])
    compressed = compress_context(context, query_text)
    model, model_type = choose_model(query_text)
    answer = ask_llm(query_text, compressed)
    return QueryResponse(
        answer=answer,
        model_used=model_type
    )