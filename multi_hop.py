from retriever import retrieve
from query_rewriter import rewrite_query

def multi_hop_retrieve(query: str, hops: int = 2) -> list:
    all_chunks = []
    seen_texts = set()
    
    current_query = query
    
    for hop in range(hops):
        print(f"Hop {hop + 1}/{hops}: {current_query[:80]}...")
        
        # Retrieve
        chunks = retrieve(current_query, top_k=3)
        
        # أضف chunks جديدة فقط
        new_chunks = []
        for chunk in chunks:
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                all_chunks.append(chunk)
                new_chunks.append(chunk)
        
        if not new_chunks:
            break
            
        # اعمل سؤال جديد من النتائج
        if hop < hops - 1:
            context_so_far = " ".join([c["text"][:200] for c in new_chunks])
            current_query = rewrite_query(
                f"Based on: {context_so_far[:300]}\nFind more details about: {query}"
            )
    
    print(f"Multi-hop: {len(all_chunks)} chunks total ✅")
    return all_chunks