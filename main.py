from dotenv import load_dotenv
from context_builder import build_context
from llm import ask_llm
from query_rewriter import rewrite_query
from reranker import rerank
from multi_hop import multi_hop_retrieve
from context_compressor import compress_context

load_dotenv()

query = "Derive the Hamiltonian for X(4) critical point symmetry"

print("=== Layer 2: Query Rewriting ===")
rewritten_query = rewrite_query(query)

print("\n=== Layer 3: Multi-hop Retrieval ===")
chunks = multi_hop_retrieve(rewritten_query, hops=2)

print("\n=== Layer 2: Re-ranking ===")
reranked = rerank(rewritten_query, chunks)

print("\n=== Layer 1: Context Builder ===")
context = build_context(reranked[:4])

print("\n=== Layer 4: Context Compression ===")
compressed_context = compress_context(context, query)

print("\n=== Layer 4: Model Routing + LLM ===")
answer = ask_llm(query, compressed_context)

print(f"\nالإجابة:\n{answer}")