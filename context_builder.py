def build_context(chunks: list) -> str:
    # Structure + Filter + Prioritize
    seen = set()
    filtered = []
    
    for chunk in chunks:
        text = chunk["text"].strip()
        # Filter تكرار
        if text not in seen:
            seen.add(text)
            filtered.append(chunk)
    
    # Sort by score (Prioritize)
    filtered.sort(key=lambda x: x["score"], reverse=True)
    
    # Structure
    context = ""
    for i, chunk in enumerate(filtered):
        context += f"[Source {i+1} - Score: {chunk['score']:.3f}]\n"
        context += f"{chunk['text']}\n\n"
    
    return context.strip()