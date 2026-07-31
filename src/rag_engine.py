# src/rag_engine.py
import os
from dotenv import load_dotenv
from google import genai
from vector_store import get_collection, search
from rag_prompt import RAG_SYSTEM_PROMPT, build_rag_prompt

# 1. Load environment variables from .env file FIRST
load_dotenv()

# 2. Get GEMINI_API_KEY from environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("[ERROR] GEMINI_API_KEY not found! Check your .env file.")

# 3. Initialize Gemini client explicitly with key
client = genai.Client(api_key=api_key)

def query_llm(system_prompt: str, user_prompt: str, model_name: str = "gemini-3.1-flash-lite") -> str:
    """Sends system and user prompts to Gemini."""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.0,  # Zero temperature for strictly grounded answers
            }
        )
        return response.text
    except Exception as e:
        return f"[LLM ERROR] Failed to generate answer: {str(e)}"


def ask(query: str, collection_name: str = "knowledge_base", top_k: int = 3, filter_source: str = None) -> dict:
    """
    Complete RAG Pipeline Execution:
    Search DB -> Augment Prompt -> Generate LLM Answer.
    """
    collection = get_collection(collection_name)
    
    # 1. Prepare optional metadata filter (e.g., filter to a specific document)
    where_filter = {"source": filter_source} if filter_source else None
    
    # 2. Retrieve top_k chunks from ChromaDB
    retrieved_results = search(collection, query=query, top_k=top_k, filter_metadata=where_filter)
    
    if not retrieved_results:
        return {
            "answer": "I do not have any relevant documents stored to answer this question.",
            "retrieved_chunks": []
        }
        
    # 3. Build Augmented RAG Prompt
    augmented_prompt = build_rag_prompt(query, retrieved_results)
    
    # 4. Generate Grounded Answer from LLM
    answer = query_llm(RAG_SYSTEM_PROMPT, augmented_prompt)
    
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_results,
        "augmented_prompt": augmented_prompt
    }


if __name__ == "__main__":
    print("RAG Engine module initialized successfully with Gemini!")