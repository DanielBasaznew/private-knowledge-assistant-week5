# src/rag_prompt.py

RAG_SYSTEM_PROMPT = """You are a precise, helpful technical AI assistant.
Your job is to answer user questions STRICTLY using only the provided context chunks below.

CRITICAL RULES:
1. ONLY use information explicitly stated in the CONTEXT block below. Do NOT use outside knowledge or make assumptions.
2. IF the provided context does NOT contain enough information to answer the question, clearly state: "I do not have enough information in the provided document(s) to answer this question."
3. Always cite the specific chunk ID or source/page number that supports your claims (e.g., [Chunk 1, Page 2]).
"""

def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Formats retrieved ChromaDB search results into a clean, labeled context block
    and appends the user's query.
    """
    formatted_context_blocks = []
    
    for idx, item in enumerate(retrieved_chunks, start=1):
        doc_text = item.get("document", "").strip()
        metadata = item.get("metadata", {})
        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page", "N/A")
        
        block = (
            f"--- CHUNK {idx} ---\n"
            f"Source: {source} (Page {page})\n"
            f"Text:\n{doc_text}"
        )
        formatted_context_blocks.append(block)
        
    full_context_str = "\n\n".join(formatted_context_blocks)
    
    user_prompt = (
        f"CONTEXT INFORMATION:\n"
        f"=====================\n"
        f"{full_context_str}\n"
        f"=====================\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Answer the question using ONLY the context provided above. If the context is insufficient, state that you do not have enough information."
    )
    
    return user_prompt

# Smoke test when executing directly
if __name__ == "__main__":
    test_chunks = [
        {
            "document": "The Python language was released in 1991 by Guido van Rossum.",
            "metadata": {"source": "python_history.pdf", "page": 1}
        },
        {
            "document": "Guido van Rossum served as Python's Benevolent Dictator For Life until 2018.",
            "metadata": {"source": "python_history.pdf", "page": 2}
        }
    ]
    
    test_prompt = build_rag_prompt("Who created Python?", test_chunks)
    print("--- SAMPLE GENERATED RAG PROMPT ---")
    print(test_prompt)