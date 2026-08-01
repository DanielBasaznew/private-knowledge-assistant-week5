# 🧠 Private Knowledge Assistant (Multi-Document RAG CLI)

A full-stack, local **Retrieval-Augmented Generation (RAG)** application built from scratch in Python. This assistant allows users to ingest PDFs, Markdown files, and plain text documents into a persistent **ChromaDB** vector database, perform semantic similarity searches, and generate grounded, cited answers using Google Gemini.

Unlike generic black-box wrappers, this project implements custom token/paragraph chunking, metadata tagging, context-window guardrails, and **Federated Multi-Source Retrieval** to eliminate vector dominance across multi-document collections.

---

## ✨ Key Features

* **Multi-Format Ingestion:** Ingests `.pdf` (via PyMuPDF), `.txt`, and `.md` documents with automated metadata tagging (Source Filename, Page Number, File Type).
* **Transparent Source Citations:** Displays every retrieved chunk—including Source Filename, Page Number, and Cosine Distance—before presenting the generated answer.
* **Three Flexible Retrieval Modes:**
  * **Global Search (`<question>`):** Queries all loaded documents simultaneously.
  * **Source Filtering (`filter <source> <question>`):** Pre-filters the vector database to search strictly within one specific document.
  * **Federated Synthesis (`compare <src1> <src2> <question>`):** Executes balanced multi-source retrieval (`top_k=6` per source) to prevent large or dense documents from starving smaller documents of context representation.
* **Production Guardrails:**
  * **Zero-Hallucination Prompting:** Enforces strict grounding with zero-temperature LLM generation and explicit "I don't know" fallback behavior.
  * **Context Window Protector:** `trim_context_if_needed()` caps context size to prevent exceeding token limits.
* **Structure-Aware Chunking:** Uses `tiktoken` sliding-window token chunking with paragraph boundary preservation and configurable overlap.

---

## 📁 Repository Structure

```text
private-knowledge-assistant-week5/
├── .env.example                # Template for environment variables (API Keys)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── main.py                     # Entrypoint script placeholder
├── test_setup.py               # Verification script for environment & dependencies
├── data/
│   ├── chroma_db/              # Persistent ChromaDB vector database storage
│   └── raw_docs/               # Sample document repository (.pdf, .txt, .md)
│       ├── frankenstein.txt
│       ├── my_notes.md
│       └── sample_paper.pdf
└── src/
    ├── knowledge_assistant.py  # Interactive CLI Application entrypoint
    ├── chunking.py             # Token & paragraph chunking logic (Tiktoken)
    ├── ingestion.py            # PDF & text file parsing & vector store ingestion
    ├── vector_store.py         # ChromaDB client initialization & similarity search
    ├── rag_prompt.py           # Grounding system prompt & prompt augmentation
    ├── rag_engine.py           # Gemini LLM generation engine
    ├── embeddings.py           # SentenceTransformers embedding function wrapper
    ├── CHUNK_EXPERIMENT.md     # Quantitative & qualitative chunk size study
    └── Journal.md              # Daily engineering log & architectural decisions
```

---


## 🛠️ Setup & Installation

### 1. Prerequisites
* **Python 3.10+** installed on your system.
* A **Google Gemini API Key** (obtainable from Google AI Studio).

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/private-knowledge-assistant-week5.git
cd private-knowledge-assistant-week5
```

### 3. Create and Activate a Virtual Environment

* **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration
Create a `.env` file in the root directory and add your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 6. Run the Assistant CLI
```bash
python src/knowledge_assistant.py
```

---

## 💻 CLI Commands & Usage

| Command | Description | Example |
| :--- | :--- | :--- |
| `load <path>` | Ingests a PDF, TXT, or MD file into ChromaDB | `load data/raw_docs/sample_paper.pdf` |
| `list` | Displays a Rich table of loaded documents & chunk counts | `list` |
| `filter <src> <q>` | Queries strictly within a single document | `filter sample_paper.pdf What is Attention?` |
| `compare <src1> <src2> <q>` | Performs balanced synthesis across two documents | `compare sample_paper.pdf my_notes.md How is attention described?` |
| `<your question>` | Performs broad semantic search across all loaded documents | `What are the core concepts discussed?` |
| `exit` | Quits the interactive CLI assistant | `exit` |

---

## 🌟 Example Interactive Session

### Cross-Document Synthesis (`compare`)

```text
Assistant > compare sample_paper.pdf my_notes.md How are attention mechanisms described in the paper compared to my notes?

================ BALANCED RETRIEVED CONTEXT ================ 
Chunk 1 | Source: sample_paper.pdf (Page 3) | Distance: 0.491
Chunk 2 | Source: sample_paper.pdf (Page 6) | Distance: 0.534
Chunk 3 | Source: sample_paper.pdf (Page 4) | Distance: 0.589
Chunk 4 | Source: my_notes.md (Page 1)      | Distance: 0.787
============================================================

CROSS-DOCUMENT SYNTHESIS ANSWER:
The provided documents describe attention mechanisms as follows:

* In the paper (sample_paper.pdf): Attention is described as a mapping from a query and key-value pairs to an output vector, calculated via weighted sums. Scaled dot-product attention uses a 1/sqrt(d_k) factor to prevent small gradients.
* In your notes (my_notes.md): Attention is noted as the core mechanism allowing sequence processing in parallel, forming the foundation of modern Transformer architectures like Gemini.
```

---

## 🔬 Scientific Chunking Experiment

To evaluate how chunk size affects vector retrieval resolution and context fragmentation, an experiment was conducted comparing **200**, **500**, and **1000** token chunks on a full literary text (*Frankenstein*).

| Chunk Size | Overlap | Chunks Generated | Precision | Failure Mode / Retained Context |
| :---: | :---: | :---: | :--- | :--- |
| **200 Tokens** | 40 Tokens | ~433 | High | **Context Fragmentation:** Slices narrative thoughts mid-sentence. |
| **500 Tokens** | 50 Tokens | ~180 | Optimal (**Goldilocks**) | **Balanced Context:** Preserves complete paragraphs and multi-sentence ideas. |
| **1000 Tokens** | 100 Tokens | ~90 | Low | **Vector Dilution:** Dense vectors average out multiple topics, missing specific queries. |

👉 For full methodology, metrics, and recommendations, see [src/CHUNK_EXPERIMENT.md](src/CHUNK_EXPERIMENT.md).  
👉 For daily development logs and architectural choices, see [src/Journal.md](src/Journal.md).  


---

## ⚙️ Tech Stack

* **Language:** Python 3.10+
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (Persistent Vector Store)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **LLM Engine:** Google Gemini API (`google-genai`)
* **Parsing & Tokenization:** `tiktoken`, `PyMuPDF` (`fitz`)
* **Terminal UI:** `rich`