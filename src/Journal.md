## Day 1: Embeddings & Similarity Search

**Key Observations:**

* **Math Reflects Reality:** The high similarity score (0.506) between the ML and Neural Network sentences perfectly mirrors how closely those concepts are tied together. Building predictive models and configuring API assistants previously made it very satisfying to see that relationship proven out in the vector math.
* **Model Scaling Varies:** The dog/puppy pair scored 0.537, lower than the expected ~0.85+. This is a great reminder that different models (like `all-MiniLM-L6-v2`) scale distances differently; relative ranking matters more than the absolute number.
* **Statistical Quirks:** "Machine Learning" and "Pizza" surprisingly scored higher (0.128) than "Stock Market" and "Pizza" (0.001). It highlights that embeddings rely on raw text co-occurrences, which can sometimes produce non-human logic or statistical noise.