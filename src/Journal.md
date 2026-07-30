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