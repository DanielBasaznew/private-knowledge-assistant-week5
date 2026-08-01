import sys
sys.path.append("src")
from vector_store import client, search
from ingestion import ingest_pdf

try:
    client.delete_collection("test_paper")
except:
    pass

ingest_pdf("data/raw_docs/sample_paper.pdf", collection_name="test_paper", chunk_size=350, overlap=50)
col = client.get_collection("test_paper")

queries = [
    "What methodology did the authors use?",
    "Summarize the conclusions of the paper"
]

for q in queries:
    print(f"\n==================== QUERY: {q} ====================")
    res = search(col, q, top_k=5)
    for idx, r in enumerate(res, start=1):
        print(f"--- CHUNK {idx} | Page {r['metadata'].get('page')} | Dist {r['distance']:.3f} ---")
        print(r['document'][:250].replace("\n", " "))
