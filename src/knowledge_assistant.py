# src/knowledge_assistant.py
import os
import sys
from rich.console import Console
from rich.table import Table

# Import our custom modules built throughout the week
from vector_store import get_collection, search
from ingestion import ingest_pdf, ingest_text_file
from rag_prompt import RAG_SYSTEM_PROMPT, build_rag_prompt
from rag_engine import query_llm

console = Console()
COLLECTION_NAME = "private_knowledge_base"

def trim_context_if_needed(retrieved_chunks: list, max_chars: int = 12000) -> list:
    """
    Context Window Protection Guardrail:
    Iterates through ranked chunks and includes them only until max_chars threshold is reached.
    """
    trimmed_chunks = []
    total_chars = 0

    for chunk in retrieved_chunks:
        # Check both possible key names depending on your vector store schema
        text = chunk.get("text") or chunk.get("document", "")
        if total_chars + len(text) > max_chars and trimmed_chunks:
            console.print(f"[dim yellow]⚠ Context limit reached ({total_chars} chars). Trimming lower-ranked chunks.[/dim yellow]")
            break
        trimmed_chunks.append(chunk)
        total_chars += len(text)

    return trimmed_chunks

class KnowledgeAssistant:
    def __init__(self):
        """Initializes the persistent ChromaDB collection."""
        self.collection = get_collection(COLLECTION_NAME)
        console.print("[bold green]Knowledge Assistant initialized successfully![/bold green]")

    def load_document(self, file_path: str):
        """Detects file type and ingests PDF or TXT/MD files into the database."""
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File not found at '{file_path}'")
            return

        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        with console.status(f"[bold yellow]Ingesting '{filename}'...[/bold yellow]", spinner="dots"):
            try:
                if ext == ".pdf":
                    chunks_added = ingest_pdf(file_path, collection_name=COLLECTION_NAME)
                elif ext in [".txt", ".md"]:
                    chunks_added = ingest_text_file(file_path, collection_name=COLLECTION_NAME)
                else:
                    console.print(f"[bold red]Unsupported file type:[/bold red] '{ext}'. Only .pdf, .txt, and .md are supported.")
                    return
                
                console.print(f"[bold green]✓ Successfully ingested '{filename}' ({chunks_added} chunks).[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error ingesting file:[/bold red] {str(e)}")

    def list_documents(self):
        """Retrieves and displays a clean Rich table of all unique loaded documents."""
        results = self.collection.get()
        metadatas = results.get("metadatas", [])

        if not metadatas:
            console.print("[yellow]No documents currently loaded in the knowledge base.[/yellow]")
            return

        # Aggregate counts by document source
        doc_counts = {}
        for meta in metadatas:
            source = meta.get("source", "Unknown")
            file_type = meta.get("file_type", "N/A")
            if source not in doc_counts:
                doc_counts[source] = {"count": 0, "type": file_type}
            doc_counts[source]["count"] += 1

        # Build Rich Table
        table = Table(title="Loaded Documents in Knowledge Base")
        table.add_column("Source Filename", style="cyan", no_wrap=True)
        table.add_column("File Type", style="magenta")
        table.add_column("Total Chunks", style="green", justify="right")

        for source, info in doc_counts.items():
            table.add_row(source, info["type"].upper(), str(info["count"]))

        console.print(table)
        console.print(f"[bold]Total Chunks Stored:[/bold] {len(metadatas)}\n")

    def ask(self, query: str, filter_source: str = None, top_k: int = 6):
        """
        Executes full multi-document RAG pipeline with visible transparency of retrieved sources.
        """
        where_filter = {"source": filter_source} if filter_source else None

        with console.status("[bold yellow]Searching knowledge base & generating grounded answer...[/bold yellow]"):
            retrieved_chunks = search(
                self.collection,
                query=query,
                top_k=top_k,
                filter_metadata=where_filter
            )

            if not retrieved_chunks:
                console.print("\n[bold red]No relevant context found in database.[/bold red]\n")
                return

            # Build RAG prompt and call Gemini
            retrieved_chunks = trim_context_if_needed(retrieved_chunks, max_chars=12000)
            augmented_prompt = build_rag_prompt(query, retrieved_chunks)
            answer = query_llm(RAG_SYSTEM_PROMPT, augmented_prompt)

        # 1. Print Transparency Block: Display retrieved chunks & metadata
        console.print("\n[bold cyan]================ RETRIEVED CONTEXT ================ [/bold cyan]")
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page", "N/A")
            dist = chunk.get("distance", 0.0)
            console.print(f"[dim]Chunk {idx} | Source: [bold]{source}[/bold] (Page {page}) | Distance: {dist:.3f}[/dim]")
        console.print("[bold cyan]==================================================[/bold cyan]\n")

        # 2. Print Grounded Answer
        console.print("[bold green]GROUNDED ANSWER:[/bold green]")
        console.print(answer)
        console.print("\n")

    def compare(self, source1: str, source2: str, query: str, top_k_each: int = 3):
        """
        Executes Balanced Multi-Source Retrieval by fetching top chunks from TWO distinct sources
        and combining them into a single RAG prompt for cross-document synthesis.
        """
        with console.status(f"[bold yellow]Performing balanced retrieval across '{source1}' and '{source2}'...[/bold yellow]"):
            chunks1 = search(self.collection, query=query, top_k=top_k_each, filter_metadata={"source": source1})
            chunks2 = search(self.collection, query=query, top_k=top_k_each, filter_metadata={"source": source2})
            
            combined_chunks = chunks1 + chunks2

            if not combined_chunks:
                console.print("\n[bold red]No relevant context found in either source.[/bold red]\n")
                return

            # Build unified prompt and generate synthesis
            combined_chunks = trim_context_if_needed(combined_chunks, max_chars=12000)  
            augmented_prompt = build_rag_prompt(query, combined_chunks)
            answer = query_llm(RAG_SYSTEM_PROMPT, augmented_prompt)

        # Print Transparency Block showing chunks from BOTH sources
        console.print("\n[bold cyan]================ BALANCED RETRIEVED CONTEXT ================ [/bold cyan]")
        for idx, chunk in enumerate(combined_chunks, start=1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page", "N/A")
            dist = chunk.get("distance", 0.0)
            console.print(f"[dim]Chunk {idx} | Source: [bold]{source}[/bold] (Page {page}) | Distance: {dist:.3f}[/dim]")
        console.print("[bold cyan]============================================================[/bold cyan]\n")

        # Print Grounded Answer
        console.print("[bold green]CROSS-DOCUMENT SYNTHESIS ANSWER:[/bold green]")
        console.print(answer)
        console.print("\n")

    def run(self):
        """Interactive CLI Command Loop."""
        console.print("\n[bold blue]--- Private Knowledge Assistant CLI ---[/bold blue]")
        console.print("Commands:")
        console.print("  [cyan]load <path/to/file>[/cyan]            - Ingest a PDF, TXT, or MD file")
        console.print("  [cyan]list[/cyan]                        - View all loaded documents")
        console.print("  [cyan]filter <source> <q>[/cyan]          - Ask a question filtered to ONE specific file")
        console.print("  [cyan]compare <src1> <src2> <q>[/cyan]    - Balanced cross-document synthesis across TWO files")
        console.print("  [cyan]exit[/cyan]                        - Quit the assistant")
        console.print("  [cyan]<your question>[/cyan]              - Ask any question across your entire knowledge base\n")

        while True:
            try:
                user_input = console.input("[bold yellow]Assistant > [/bold yellow]").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    console.print("[bold red]Goodbye![/bold red]")
                    break
                elif user_input.lower() == "list":
                    self.list_documents()
                elif user_input.lower().startswith("load "):
                    file_path = user_input[5:].strip()
                    self.load_document(file_path)
                elif user_input.lower().startswith("filter "):
                    parts = user_input[7:].strip().split(" ", 1)
                    if len(parts) < 2:
                        console.print("[bold red]Usage:[/bold red] filter <filename> <your question>")
                    else:
                        self.ask(parts[1], filter_source=parts[0])
                elif user_input.lower().startswith("compare "):
                    # Syntax: compare <source1> <source2> <question>
                    parts = user_input[8:].strip().split(" ", 2)
                    if len(parts) < 3:
                        console.print("[bold red]Usage:[/bold red] compare <source1> <source2> <your question>")
                    else:
                        self.compare(parts[0], parts[1], parts[2])
                else:
                    self.ask(user_input)
            except KeyboardInterrupt:
                console.print("\n[bold red]Goodbye![/bold red]")
                break

if __name__ == "__main__":
    assistant = KnowledgeAssistant()
    assistant.run()