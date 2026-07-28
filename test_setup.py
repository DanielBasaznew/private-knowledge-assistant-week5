import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import tiktoken
from rich import print

def verify_environment():
    print("[bold green]Testing dependencies...[/bold green]")
    
    # Test Tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode("Testing RAG setup")
    print(f"[green]✓[/green] Tiktoken working (tokens: {len(tokens)})")
    
    # Test SentenceTransformers (downloads all-MiniLM-L6-v2 on first run)
    print("Loading embedding model 'all-MiniLM-L6-v2' (may take a moment to download)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode("Hello RAG")
    print(f"[green]✓[/green] Embeddings working (dimension: {len(vec)})")
    
    # Test ChromaDB
    client = chromadb.Client()
    collection = client.create_collection("test_collection")
    collection.add(
        documents=["RAG pipeline test chunk"],
        ids=["id1"]
    )
    res = collection.query(query_texts=["RAG"], n_results=1)
    print(f"[green]✓[/green] ChromaDB working (retrieved: '{res['documents'][0][0]}')")

    print("\n[bold cyan]Environment is 100% ready for Week 5![/bold cyan]")

if __name__ == "__main__":
    verify_environment()
