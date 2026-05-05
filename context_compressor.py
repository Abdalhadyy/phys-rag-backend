from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def compress_context(context: str, query: str) -> str:
    prompt = f"""You are an expert in nuclear physics.
Given the following context from research papers, extract ONLY the most relevant information to answer the question.
Keep equations, numbers, and key terms intact.
Be concise but complete.

Query: {query}

Context:
{context}

Compressed context (keep only what's relevant):"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    compressed = response.choices[0].message.content.strip()
    
    original_len = len(context.split())
    compressed_len = len(compressed.split())
    print(f"Context: {original_len} words → {compressed_len} words ✅")
    
    return compressed