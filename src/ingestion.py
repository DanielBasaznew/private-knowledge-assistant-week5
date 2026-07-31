# src/ingestion.py
import os
from pypdf import PdfReader
from vector_store import get_collection, add_documents
from chunking import chunk_by_tokens, chunk_by_paragraph

BATCH_SIZE = 100  # Safe batch size to avoid memory strain or SQLite limits

def ingest_pdf(file_path: str, collection_name: str = "knowledge_base", chunk_size: int = 500, overlap: int = 50):
    """
    Extracts text from a PDF page by page, chunks each page's content,
    attaches page metadata, and upserts chunks into ChromaDB in batches.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    filename = os.path.basename(file_path)
    reader = PdfReader(file_path)
    collection = get_collection(collection_name)
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    total_pages = len(reader.pages)
    print(f"\n[INGESTING PDF] '{filename}' ({total_pages} pages)...")
    
    chunk_counter = 0
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue  # Skip blank pages
            
        # Chunk text from this specific page
        page_chunks = chunk_by_tokens(page_text, chunk_size=chunk_size, overlap=overlap)
        
        for c in page_chunks:
            chunk_id = f"{filename}_p{page_idx}_c{chunk_counter}"
            all_chunks.append(c)
            all_metadatas.append({
                "source": filename,
                "page": page_idx,
                "file_type": "pdf"
            })
            all_ids.append(chunk_id)
            chunk_counter += 1
            
    # Batch insertion into ChromaDB
    total_chunks = len(all_chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_meta = all_metadatas[i:i + BATCH_SIZE]
        batch_ids = all_ids[i:i + BATCH_SIZE]
        add_documents(collection, batch_chunks, batch_meta, batch_ids)
        
    print(f"[SUCCESS] Ingested {total_chunks} chunks from PDF '{filename}' into '{collection_name}'.\n")
    return total_chunks


def ingest_text_file(file_path: str, collection_name: str = "knowledge_base", max_chunk_size: int = 500, overlap: int = 50):
    """
    Ingests plain text files using paragraph-aware chunking.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at: {file_path}")
        
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    collection = get_collection(collection_name)
    chunks = chunk_by_paragraph(full_text, max_chunk_size=max_chunk_size, overlap=overlap)
    
    metadatas = [{"source": filename, "page": 1, "file_type": "txt"} for _ in chunks]
    ids = [f"{filename}_c{i}" for i in range(len(chunks))]
    
    # Batch insertion
    total_chunks = len(chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        add_documents(
            collection, 
            chunks[i:i + BATCH_SIZE], 
            metadatas[i:i + BATCH_SIZE], 
            ids[i:i + BATCH_SIZE]
        )
        
    print(f"[SUCCESS] Ingested {total_chunks} chunks from TXT '{filename}' into '{collection_name}'.\n")
    return total_chunks


if __name__ == "__main__":
    # Quick syntax test
    print("Ingestion engine initialized successfully.")