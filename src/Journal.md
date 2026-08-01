# 📓 Week 5 Journal: Building a Private Knowledge Assistant (Multi-Document RAG)

---

## Day 1: Embeddings & Similarity Search

**Key Observations:**
* **Math Reflects Reality:** The high similarity score (0.506) between the ML and Neural Network sentences perfectly mirrors how closely those concepts are tied together. Building predictive models and configuring API assistants previously made it very satisfying to see that relationship proven out in the vector math.
* **Model Scaling Varies:** The dog/puppy pair scored 0.537, lower than the expected ~0.85+. This is a great reminder that different models (like `all-MiniLM-L6-v2`) scale distances differently; relative ranking matters more than the absolute number.
* **Statistical Quirks:** "Machine Learning" and "Pizza" surprisingly scored higher (0.128) than "Stock Market" and "Pizza" (0.001). It highlights that embeddings rely on raw text co-occurrences, which can sometimes produce non-human logic or statistical noise.

---

## Day 2: ChromaDB Setup & Persistent Vector DB

**Key Observations:**
* **Abstraction & Automation:** ChromaDB eliminates manual matrix math by automatically embedding texts in the background via `SentenceTransformerEmbeddingFunction` during both insertion and query phases.
* **Persistence & Upsert Safety:** Using `chromadb.PersistentClient` ensures data persists on disk in `data/chroma_db/`. Switching from `.add()` to `.upsert()` prevents crash errors when re-running scripts with existing IDs.
* **Distance vs. Similarity:** ChromaDB returns Cosine Distance (0.0 = identical), which requires a `1 - distance` transformation to output intuitive similarity scores where 1.0 represents an exact match.
* **Metadata Foundation:** Attaching structured dictionaries to documents unlocks future filtered queries using the `where` parameter without degrading search performance.

---

## Day 3: Chunking Strategies & Token Size Experiments

**Key Observations:**
* **The Chunk Size Trade-off:** Smaller chunks (200 tokens) yield higher vector similarity (0.639) by isolating direct matches, but sacrifice surrounding narrative context. Larger chunks (1000 tokens) suffer from context dilution, dropping similarity to 0.336 due to extra noise.
* **Structure-Aware Chunking:** Paragraph-based chunking prevents breaking thoughts mid-sentence, while token-based chunking with overlap (10-20%) acts as a necessary safety net against severed boundary text.
* **Optimal Selection:** 500-token chunking hit the "sweet spot" for text retrieval, offering a balanced trade-off between semantic precision and sufficient LLM context.
* **Persistence Management:** Database assets persist safely on disk in `./data/chroma_db/`, requiring proper `client.delete_collection()` calls for clean resets instead of manual file deletion.

---

## Day 4: Full RAG Pipeline & Grounded Generation

**Key Observations:**
* **End-to-End Modular Architecture:** Integrated document ingestion (`pypdf` + paragraph chunking), persistent vector retrieval (ChromaDB), prompt augmentation, and Gemini generation into a unified RAG engine.
* **Grounding Guardrails & Zero Hallucinations:** Enforcing strict prompt rules (`temperature=0.0`, explicit instructions to admit ignorance, and chunk-level citation requirements) prevented the LLM from relying on outside pre-training knowledge.
* **Retrieval Boundaries:** When asking specific questions (e.g., Chapter 1 themes or paper methodology), returning `top_k=3` chunks meant the model correctly refused to answer if the target content was outside those 3 chunks, proving prompt grounding works as designed.
* **RAG vs. Fine-Tuning:** RAG remains the optimal pattern for dynamic, verifiable knowledge retrieval with explicit source attribution, whereas fine-tuning is better suited for style and format customization.

---

## Day 5: Multi-Document RAG & The Private Knowledge Assistant

### 1. Architecture & Unified Interface
* Built a complete, interactive CLI application (`knowledge_assistant.py`) wrapping ingestion, vector retrieval, prompt grounding, and LLM generation into a single `KnowledgeAssistant` class.
* Integrated `rich` to provide visual terminal feedback (loading spinners, error formatting) and a structured summary table showing unique source filenames, media types (PDF, TXT, MD), and total chunk counts.
* Implemented full retrieval transparency: every answer displays the `RETRIEVED CONTEXT` block (Chunk ID, Source Filename, Page Number, Cosine Distance) before printing the LLM's response, allowing immediate verification of grounding.

### 2. Global Semantic Search vs. Source Filtering
* **Global Search (`<question>`):** Evaluated semantic similarity routing across a multi-document database (463 total chunks: 29 PDF, 433 TXT, 1 MD).
  * *Observation:* Distinct domain vocabulary (e.g., "AI agents", "8 GPUs", "Beaufort") allows cosine similarity to naturally surface the correct document to Chunk 1 without manual intervention.
  * *Limitation:* When the primary target document has fewer chunks than `top_k=6` (e.g., a 1-chunk markdown note), ChromaDB fills remaining slots with mathematically closest chunks from other documents. Zero-temperature grounding prompts successfully instruct the LLM to ignore this out-of-domain noise.
* **Source Filtering (`filter <source> <question>`):** Implemented database-level pre-filtering (`where={"source": filename}`) to restrict similarity searches strictly to a single document.
  * *Result:* Eliminates cross-contamination entirely, ensuring precision for deep technical or literary extraction.

### 3. Resolving Vector Dominance via Federated Retrieval
* **The Engineering Problem ("Needle in a Haystack"):** In broad cross-document queries (e.g., comparing attention mechanisms), large vocabulary-dense documents (the 29-chunk academic paper) crowd out smaller documents (the 1-chunk personal note) from the `top_k=6` window.
* **The Solution:** Built a Federated/Balanced Retrieval command (`compare <src1> <src2> <question>`) that executes independent vector searches against two specified sources (`top_k=3` each) before merging the results into a balanced prompt.
* **The Result:** Successfully eliminated source starvation. The LLM synthesized mathematical definitions from an academic arXiv PDF and conceptual definitions from personal markdown notes into a single, accurately cited comparative response.

### 4. Grounding Guardrail Validation
* Tested plausible out-of-domain traps (e.g., querying speech recognition performance on an NLP translation paper). The model consistently adhered to system prompt rules, refusing to hallucinate and stating when documents lacked sufficient information.

---

## Day 6: Hardening, Experimentation & Codebase Refactoring

* **Context Window Protection (`trim_context_if_needed`):** Built a dynamic token/character guardrail inside `knowledge_assistant.py` that measures total retrieved text length and safely trims lower-ranked chunks if the combined context exceeds a safe ceiling (`12,000 characters` / `~3,000 tokens`). Verified real-time trimming behavior via CLI stress tests.
* **Formal Chunk Size Experiment Report (`CHUNK_EXPERIMENT.md`):** Documented our Day 3 quantitative and qualitative trade-off study comparing 200, 500, and 1000-token chunk sizes on a full-length literary text (`frankenstein.txt`). Concluded that 500 tokens is the optimal Goldilocks zone for semantic accuracy and context preservation.
* **Codebase Clean-up:** Deleted obsolete boilerplate files (`embeddings.py`, `rag_pipeline.py`) to keep the source tree intentional and verified docstring coverage across all core modules.

---

# 🏆 Week 5 Final Summary & Architectural Reflection

### 1. Explain RAG to your Week 1 self (no undefined jargon):
In Week 1, calling an LLM API was like asking a super-smart friend a trivia question—if they hadn't read about the topic, they either guessed (hallucinated) or said they didn't know. **Retrieval-Augmented Generation (RAG)** is like giving that same friend an open-book exam. Before we ask the LLM our question, we first search a local database of our own private documents (like PDFs or notes) to find the paragraphs most relevant to the question. Then, we paste those paragraphs into the prompt alongside our question and tell the model: *"Answer my question using ONLY these facts."* This lets the LLM answer questions about private or brand-new data it was never trained on.

### 2. What is the single most important decision in building a RAG system, and why?
The single most important decision is your **Chunking Strategy** (how you slice your documents into smaller pieces before storing them). As proven in our chunk size experiment, if your chunks are too small (`200 tokens`), you fragment ideas and strip away necessary context; if they are too large (`1000 tokens`), you dilute the embedding vector—causing precise, factual queries to miss the chunk entirely during similarity search. Every downstream step (retrieval accuracy, prompt size, and LLM grounding) depends entirely on well-balanced, context-preserving chunks.

### 3. What broke most this week, and how did you fix it?
The most stubborn issue was **Vector Dominance ("Needle in a Haystack")** during multi-document queries. When querying across a 433-chunk novel, a 29-chunk academic PDF, and a 1-chunk personal markdown note using standard Global Search (`top_k=6`), the larger, vocabulary-dense documents completely crowded out the smaller note—even when the query specifically asked about concepts in the note. I fixed this by engineering a **Federated Retrieval (`compare`) command**, which independently queries ChromaDB for the top chunks from each target document before merging them into a balanced prompt window.

### 4. How do you think Week 6 will improve on what you built this week?
Week 6 ("Better Retrieval + Memory") will address the limitations of pure vector similarity search by introducing **Cross-Encoder Re-Ranking**. Instead of relying solely on fast embedding distance—which only measures general topic similarity—we will use a two-stage retrieval pipeline that re-scores candidate chunks based on how well they directly *answer* the question. Additionally, adding persistent memory will allow the assistant to track conversation context across multiple turns without overflowing the token limit.