## Day 1: Embeddings & Similarity Search

**Key Observations:**

* **Math Reflects Reality:** The high similarity score (0.506) between the ML and Neural Network sentences perfectly mirrors how closely those concepts are tied together. Building predictive models and configuring API assistants previously made it very satisfying to see that relationship proven out in the vector math.
* **Model Scaling Varies:** The dog/puppy pair scored 0.537, lower than the expected ~0.85+. This is a great reminder that different models (like `all-MiniLM-L6-v2`) scale distances differently; relative ranking matters more than the absolute number.
* **Statistical Quirks:** "Machine Learning" and "Pizza" surprisingly scored higher (0.128) than "Stock Market" and "Pizza" (0.001). It highlights that embeddings rely on raw text co-occurrences, which can sometimes produce non-human logic or statistical noise.

## Day 2: ChromaDB Setup & Persistent Vector DB

**Key Observations:**

* **Abstraction & Automation:** ChromaDB eliminates manual matrix math by automatically embedding texts in the background via `SentenceTransformerEmbeddingFunction` during both insertion and query phases.
* **Persistence & Upsert Safety:** Using `chromadb.PersistentClient` ensures data persists on disk in `data/chroma_db/`. Switching from `.add()` to `.upsert()` prevents crash errors when re-running scripts with existing IDs.
* **Distance vs. Similarity:** ChromaDB returns Cosine Distance ($0.0 = \text{identical}$), which requires a `1 - distance` transformation to output intuitive similarity scores where $1.0$ represents an exact match.
* **Metadata Foundation:** Attaching structured dictionaries to documents unlocks future filtered queries using the `where` parameter without degrading search performance.

## Day 3: Chunking Strategies & Token Size Experiments

**Key Observations:**

* **The Chunk Size Trade-off:** Smaller chunks (200 tokens) yield higher vector similarity (0.639) by isolating direct matches, but sacrifice surrounding narrative context. Larger chunks (1000 tokens) suffer from context dilution, dropping similarity to 0.336 due to extra noise.
* **Structure-Aware Chunking:** Paragraph-based chunking prevents breaking thoughts mid-sentence, while token-based chunking with overlap (10-20%) acts as a necessary safety net against severed boundary text.
* **Optimal Selection:** 500-token chunking hit the "sweet spot" for text retrieval, offering a balanced trade-off between semantic precision and sufficient LLM context.
* **Persistence Management:** Database assets persist safely on disk in `./data/chroma_db/`, requiring proper `client.delete_collection()` calls for clean resets instead of manual file deletion.

## Day 4: Full RAG Pipeline & Grounded Generation

**Key Observations:**

* **End-to-End Modular Architecture:** Integrated document ingestion (`pypdf` + paragraph chunking), persistent vector retrieval (ChromaDB), prompt augmentation, and Gemini generation into a unified RAG engine.
* **Grounding Guardrails & Zero Hallucinations:** Enforcing strict prompt rules (`temperature=0.0`, explicit instructions to admit ignorance, and chunk-level citation requirements) prevented the LLM from relying on outside pre-training knowledge.
* **Retrieval Boundaries:** When asking specific questions (e.g., Chapter 1 themes or paper methodology), returning `top_k=3` chunks meant the model correctly refused to answer if the target content was outside those 3 chunks, proving prompt grounding works as designed.
* **RAG vs. Fine-Tuning:** RAG remains the optimal pattern for dynamic, verifiable knowledge retrieval with explicit source attribution, whereas fine-tuning is better suited for style and format customization.

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