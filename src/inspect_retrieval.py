from vector_store import get_collection, search

col = get_collection("rag_knowledge_base")

queries = [
    ("What is the main theme of chapter 1?", "frankenstein.txt"),
    ("What methodology did the authors use?", "sample_paper.pdf"),
    ("Summarize the conclusions of the paper", "sample_paper.pdf"),
    ("Who are the main characters and what is their conflict?", "frankenstein.txt"),
]

for q, src in queries:
    print(f"\n==================== QUERY: {q} ({src}) ====================")
    res = search(col, q, top_k=3, filter_metadata={"source": src})
    for idx, r in enumerate(res, start=1):
        print(f"--- CHUNK {idx} | Page {r['metadata'].get('page')} | Dist {r['distance']:.3f} ---")
        print(r['document'])
