# test_day2.py
from vector_store import get_collection, add_documents, search

# 1. Connect to our persistent knowledge base
collection = get_collection("my_handcrafted_kb")

# 2. Six distinct facts to store
facts = [
    "Python was created by Guido van Rossum and released in 1991.",
    "The Amazon rainforest produces roughly 20 percent of the Earth's oxygen.",
    "Ada Lovelace is widely considered the world's first computer programmer.",
    "The English Premier League was founded in 1992 and is the most-watched football league.",
    "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    "Machine learning models use neural networks inspired by the human brain."
]

# 3. Create metadata and unique IDs for each fact
metadatas = [{"source": "handcrafted_fact", "id_num": i} for i in range(len(facts))]
ids = [f"fact_{i}" for i in range(len(facts))]

print("\n--- STEP A: Adding documents to ChromaDB ---")
add_documents(collection, facts, metadatas, ids)

print(f"Total documents now stored in collection: {collection.count()}")

# 4. Test Query 1: Exact topic keyword match
query1 = "Who created Python?"
print(f"\n--- STEP B: Searching for -> '{query1}' ---")
results1 = search(collection, query=query1, top_k=2)

for r in results1:
    print(f"ID: {r['id']} | Sim: {r['similarity']:.3f} (Dist: {r['distance']:.3f})")
    print(f"   => {r['document']}")

# 5. Test Query 2: Semantic rephrasing (testing meaning over keywords)
query2 = "Who invented the Python programming language?"
print(f"\n--- STEP C: Searching for -> '{query2}' ---")
results2 = search(collection, query=query2, top_k=2)

for r in results2:
    print(f"ID: {r['id']} | Sim: {r['similarity']:.3f} (Dist: {r['distance']:.3f})")
    print(f"   => {r['document']}")