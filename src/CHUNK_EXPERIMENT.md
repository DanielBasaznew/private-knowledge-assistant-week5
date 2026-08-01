# Chunk Size & Retrieval Quality Experiment

## 1. Objective
This experiment investigates the trade-offs between chunk size (in tokens), database storage footprint (total chunks), and retrieval accuracy in a Retrieval-Augmented Generation (RAG) pipeline.

---

## 2. Experimental Setup
* **Test Document:** `frankenstein.txt` (approx. 75,000 words / full literary text)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
* **Vector Store:** ChromaDB (Persistent Collection)
* **Overlap Strategy:** Fixed 10%–20% token overlap to preserve sentence boundaries.

---

## 3. Quantitative Results

| Chunk Size (Tokens) | Overlap (Tokens) | Total Chunks Generated | Embedding Speed | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **200 Tokens** | 40 Tokens | ~433 Chunks | Moderate | Higher (More vectors to index) |
| **500 Tokens** | 50 Tokens | ~180 Chunks | Fast | Moderate |
| **1000 Tokens** | 100 Tokens | ~90 Chunks | Very Fast | Low |

*(Note: Exact chunk counts vary slightly based on formatting and punctuation stripping).*

---

## 4. Qualitative Retrieval Observations

### A. Small Chunks (200 Tokens)
* **Precision:** High semantic specificity. Cosine similarity distance scores are very tight when matching exact terms or names.
* **Failure Mode:** **Context Fragmentation.** Often slices a character's speech or a technical explanation in half, forcing the LLM to answer without critical surrounding context (e.g., pronoun references are lost).

### B. Medium Chunks (500 Tokens)
* **Precision:** Optimal balance. Captures complete paragraphs and self-contained thoughts.
* **Failure Mode:** Occasional minor noise if a topic shifts midway through a long paragraph.
* **Verdict:** The **Goldilocks Zone** for literary and technical narrative documents.

### C. Large Chunks (1000 Tokens)
* **Precision:** Poor semantic resolution.
* **Failure Mode:** **Vector Dilution.** Because 1,000 tokens contain multiple distinct ideas, the resulting 384-dimensional embedding represents an "average" of all those ideas. Consequently, precise factual queries fail to rank these chunks in the top-$k$ results.

---

## 5. Architectural Recommendations

1. **Use 200 Tokens When:** Indexing structured, dense factual data such as FAQs, glossary definitions, or short code snippets where every sentence introduces a distinct concept.
2. **Use 500 Tokens When:** Indexing narrative documents, research papers, or books where ideas span multiple paragraphs and require surrounding context for LLM comprehension.
3. **Avoid 1000+ Tokens Unless:** Using large-window summarization pipelines or hierarchical retrieval architectures (e.g., retrieving a large parent chunk only after matching a smaller child chunk).