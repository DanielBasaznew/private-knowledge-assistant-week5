# src/embeddings_explorer.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the local, free embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The dog ran across the park",
    "A puppy sprinted through the garden",
    "Machine learning is a subset of AI",
    "Neural networks learn from data",
    "The stock market crashed yesterday",
    "Pizza is my favourite food",
]

# 1. Translate sentences into vectors
embeddings = model.encode(sentences)
print(f"Each embedding is a vector of {len(embeddings[0])} numbers")

# 2. Compute cosine similarity between all pairs
similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity scores (0 = unrelated, 1 = identical meaning):")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        # Only print unique pairs (ignore comparing a sentence to itself)
        if i < j:
            score = similarity_matrix[i][j]
            print(f"  {score:.3f} | '{s1[:35]}' vs '{s2[:35]}'")


# --- STEP 2: MANUAL VECTOR SEARCH ---

def semantic_search(query: str, documents: list, model, top_k: int = 3):
    """Find the most semantically similar documents to a query."""
    
    # 1. Convert the query into a vector
    query_embedding = model.encode([query])
    
    # 2. Convert all documents into vectors
    doc_embeddings = model.encode(documents)
    
    # 3. Measure how close the query is to each document
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # 4. Sort the results (np.argsort sorts lowest to highest, [::-1] reverses it to highest first)
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in ranked_indices:
        results.append({
            "document": documents[idx],
            "score": float(similarities[idx])
        })
    return results

# Test the search function:
search_documents = [
    "Python is a high-level programming language known for readability",
    "Cats are independent animals that sleep 16 hours a day",
    "JavaScript runs in the browser and enables interactive web pages",
    "The Eiffel Tower is located in Paris, France",
    "Machine learning models learn patterns from training data",
    "Rust is a systems programming language focused on safety",
]

query = "What programming languages are popular?"
print(f"\n--- Searching for: '{query}' ---")

results = semantic_search(query, search_documents, model)
for r in results:
    print(f"{r['score']:.3f} | {r['document']}")