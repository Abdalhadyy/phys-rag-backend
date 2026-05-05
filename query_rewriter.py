from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_query(query: str) -> str:
    prompt = f"""You are an expert in nuclear physics.
Rewrite the following question to be more specific and technical for searching in nuclear physics research papers.
Keep it concise (1-2 sentences).

Original question: {query}

Rewritten question:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    
    rewritten = response.choices[0].message.content.strip()
    print(f"السؤال الأصلي: {query}")
    print(f"السؤال بعد Rewriting: {rewritten}")
    return rewritten