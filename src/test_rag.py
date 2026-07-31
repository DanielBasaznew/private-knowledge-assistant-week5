# src/test_rag.py
import os
import urllib.request
from ingestion import ingest_pdf, ingest_text_file
from rag_engine import ask

# 1. Setup paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_docs")
txt_path = os.path.join(data_dir, "frankenstein.txt")
pdf_path = os.path.join(data_dir, "sample_paper.pdf")

# 2. Ensure sample PDF exists (Download Attention Is All You Need if missing)
if not os.path.exists(pdf_path):
    print("[INFO] Downloading sample arXiv PDF for testing...")
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    urllib.request.urlretrieve(url, pdf_path)
    print("[INFO] Downloaded sample_paper.pdf successfully.")

# 3. Ingest documents into ChromaDB
print("\n--- INGESTING DOCUMENTS ---")
ingest_text_file(txt_path, collection_name="rag_knowledge_base")
ingest_pdf(pdf_path, collection_name="rag_knowledge_base")

# 4. Define the 4 test questions
test_queries = [
    {
        "q": "What is the main theme of chapter 1?",
        "source_filter": "frankenstein.txt",
        "description": "Book Test: Early chapter theme retrieval"
    },
    {
        "q": "What methodology did the authors use?",
        "source_filter": "sample_paper.pdf",
        "description": "Paper Test: Specific methodology section"
    },
    {
        "q": "Summarize the conclusions of the paper",
        "source_filter": "sample_paper.pdf",
        "description": "Paper Test: Conclusion section retrieval"
    },
    {
        "q": "Who are the main characters and what is their conflict?",
        "source_filter": "frankenstein.txt",
        "description": "Book Test: Broad character relationship retrieval"
    }
]

# 5. Execute queries and display grounded output
print("\n==================================================")
print("RUNNING END-TO-END RAG EVALUATION")
print("==================================================")

for item in test_queries:
    print(f"\n--------------------------------------------------")
    print(f"TEST: {item['description']}")
    print(f"QUESTION: '{item['q']}'")
    print(f"FILTER SOURCE: '{item['source_filter']}'")
    print(f"--------------------------------------------------")
    
    # Run RAG ask function
    result = ask(
        query=item['q'],
        collection_name="rag_knowledge_base",
        top_k=3,
        filter_source=item['source_filter']
    )
    
    print("\n[GROUNDED ANSWER FROM GEMINI]:")
    print(result["answer"])
    
    print("\n[RETRIEVED SOURCE CHUNKS USED]:")
    for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
        meta = chunk.get("metadata", {})
        print(f"  Chunk {idx} | Source: {meta.get('source')} (Page {meta.get('page')}) | Dist: {chunk.get('distance', 0):.3f}")