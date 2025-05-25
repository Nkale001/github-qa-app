import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# 🔹 Load embeddings and text chunks from file
def load_embeddings(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    chunks = [item["text"] for item in data]
    embeddings = [item["embedding"] for item in data]
    return chunks, np.array(embeddings)

# 🔹 Load a free embedding model from Hugging Face
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return embedder.encode([text])[0]

# 🔹 Rank chunks based on similarity to the user's question
def find_top_chunks(question, chunks, embeddings, top_k=3):
    question_embedding = get_embedding(question)
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# 🔹 Load the summarization model (also free)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_chunks(chunks):
    combined_text = " ".join(chunks)
    summary = summarizer(combined_text, max_length=200, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

# 🔹 Main function to drive the flow
def main():
    # Load your file (make sure it's in the same folder)
    chunks, embeddings = load_embeddings("embeddings.json")

    while True:
        question = input("\n❓ Ask your question (or type 'exit' to quit): ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        top_chunks = find_top_chunks(question, chunks, embeddings, top_k=3)
        summary = summarize_chunks(top_chunks)

        print("\n📌 Summary:")
        print(summary)

# 🔹 Run it!
if __name__ == "__main__":
    main()
