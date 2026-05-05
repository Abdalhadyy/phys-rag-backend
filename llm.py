from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MATH_KEYWORDS = [
    "calculate", "derive", "equation", "hamiltonian", "eigenvalue",
    "integral", "matrix", "operator", "solve", "compute",
    "احسب", "اشتق", "معادلة", "حل", "اوجد",
    "R4/2", "ζ", "β", "γ", "B(E2)", "E(4+)", "E(2+)"
]

PHYSICS_KEYWORDS = [
    "explain", "describe", "what is", "summarize", "compare",
    "اشرح", "لخص", "قارن", "ما هو", "ما هي"
]

def choose_model(query: str) -> tuple:
    query_lower = query.lower()
    
    math_score = sum(1 for k in MATH_KEYWORDS if k.lower() in query_lower)
    physics_score = sum(1 for k in PHYSICS_KEYWORDS if k.lower() in query_lower)
    
    if math_score > physics_score:
        return "qwen/qwen3-32b", "Math/Derivation"
    else:
        return "llama-3.3-70b-versatile", "Physics/Analysis"

def ask_llm(query: str, context: str) -> str:
    model, model_type = choose_model(query)
    print(f"النموذج: {model_type} → {model}")

    prompt = f"""You are an expert in nuclear physics specializing in:
- Interacting Boson Model (IBM)
- Critical Point Symmetries X(3), X(4), X(5)
- Geometric Collective Model (GCM)
- Nuclear structure and collective models

Answer based on the following context:

{context}

Question: {query}

Provide a detailed and accurate scientific answer."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content