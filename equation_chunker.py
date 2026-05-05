import re
import fitz

def extract_text_with_equations(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    blocks = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        blocks.append({
            "text": text,
            "page": page_num + 1,
            "type": "text"
        })
    
    doc.close()
    return blocks

def is_equation(text: str) -> bool:
    equation_patterns = [
        r'[A-Za-z]\s*=\s*[\d\+\-\*\/\(\)]',  # x = ...
        r'R\d+\/\d+',                           # R4/2
        r'E\([0-9\+]+\)',                        # E(4+)
        r'B\(E2\)',                              # B(E2)
        r'H\s*=',                               # H =
        r'ζ|β|γ|λ|μ|σ|τ',                      # Greek letters
        r'\d+\s*[±]\s*\d+',                     # numbers with ±
    ]
    for pattern in equation_patterns:
        if re.search(pattern, text):
            return True
    return False

def equation_aware_chunk(blocks: list, chunk_size: int = 500) -> list:
    chunks = []
    current_chunk = ""
    current_type = "text"
    
    for block in blocks:
        text = block["text"]
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # هل الجملة دي معادلة؟
            if is_equation(sentence):
                # احفظ الـ chunk الحالي لو فيه حاجة
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "type": current_type,
                        "page": block["page"]
                    })
                    current_chunk = ""
                
                # المعادلة chunk مستقل
                chunks.append({
                    "text": sentence,
                    "type": "equation",
                    "page": block["page"]
                })
            else:
                current_chunk += " " + sentence
                current_type = "text"
                
                if len(current_chunk.split()) >= chunk_size:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "type": current_type,
                        "page": block["page"]
                    })
                    current_chunk = ""
    
    # الباقي
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "type": "text",
            "page": block["page"]
        })
    
    return chunks