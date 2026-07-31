# src/experiment_chunk_sizes.py
import os
from vector_store import get_collection, add_documents, search
from chunking import chunk_by_tokens

# 1. Load the raw book text
raw_file_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "raw_docs", "frankenstein.txt"
)

with open(raw_file_path, "r", encoding="utf-8") as f:
    full_text = f.read()

# 2. Trim Gutenberg legal boilerplate (approx 5000 chars from top and bottom)
clean_text = full_text[5000:-5000]
print(f"[INFO] Loaded Frankenstein: {len(clean_text)} characters after trimming boilerplate.\n")

# 3. Define the three configurations to test
configs = [
    {"size": 200, "overlap": 20, "collection_name": "exp_200_tokens"},
    {"size": 500, "overlap": 50, "collection_name": "exp_500_tokens"},
    {"size": 1000, "overlap": 100, "collection_name": "exp_1000_tokens"},
]

print("--- BUILDING COLLECTIONS ---")
for cfg in configs:
    collection = get_collection(cfg["collection_name"])
    
    # Generate chunks for this specific size
    chunks = chunk_by_tokens(clean_text, chunk_size=cfg["size"], overlap=cfg["overlap"])
    
    # Prepare metadata & IDs
    metadatas = [{"chunk_size": cfg["size"], "chunk_index": i} for i in range(len(chunks))]
    ids = [f"chunk_{cfg['size']}_{i}" for i in range(len(chunks))]
    
    # Add to ChromaDB
    add_documents(collection, chunks, metadatas, ids)
    print(f"-> Collection '{cfg['collection_name']}': Stored {len(chunks)} chunks.")

# 4. Run the experiment with a single meaningful query
test_query = "What created the monster and what drives Victor Frankenstein?"
print(f"\n==================================================")
print(f"RUNNING EXPERIMENTAL QUERY: '{test_query}'")
print(f"==================================================")

for cfg in configs:
    collection = get_collection(cfg["collection_name"])
    results = search(collection, query=test_query, top_k=1)
    
    print(f"\n--------------------------------------------------")
    print(f"RESULTS FOR CHUNK SIZE: {cfg['size']} TOKENS")
    print(f"--------------------------------------------------")
    if results:
        top_res = results[0]
        print(f"Similarity Score: {top_res['similarity']:.3f} | Distance: {top_res['distance']:.3f}")
        print(f"Retrieved Chunk Content:\n")
        print(f"\"{top_res['document']}\"")