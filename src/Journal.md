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