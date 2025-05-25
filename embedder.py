from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Load the local model
model = SentenceTransformer('all-MiniLM-L6-v2')

# === 1. Chunking Function ===
def chunk_text(text, max_words=150):
    """Break text into smaller chunks of ~150 words."""
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# === 2. Embedding Function ===
def embed_chunks(chunks):
    """Generate embeddings locally using Hugging Face model."""
    vectors = model.encode(chunks)
    embedded = [{"text": chunk, "embedding": vector.tolist()} for chunk, vector in zip(chunks, vectors)]
    return embedded

# === 3. Test Runner ===
if __name__ == "__main__":
    with open("readme_cache.txt", "r", encoding="utf-8") as f:
        readme = f.read()

    chunks = chunk_text(readme)
    print(f"✅ Split into {len(chunks)} chunks.")

    embedded_data = embed_chunks(chunks)
    print(f"✅ Generated embeddings for {len(embedded_data)} chunks.")

    # Save the embeddings to a JSON file
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embedded_data, f)

    print("📦 Embeddings saved to embeddings.json.")
