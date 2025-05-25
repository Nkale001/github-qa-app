import streamlit as st
import requests
import torch
from transformers import pipeline

st.cache_data.clear()

def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_pipeline()

def fetch_github_readme(repo_url):
    try:
        repo_owner, repo_name = repo_url.split("https://github.com/")[1].split("/")[:2]
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
        response = requests.get(api_url)
        if response.status_code != 200:
            return None
        files = response.json()
        for file in files:
            if file["name"].lower() == "readme.md":
                readme_resp = requests.get(file["download_url"])
                if readme_resp.status_code == 200:
                    return readme_resp.text
        return None
    except Exception:
        return None

st.title("GitHub Repo Q&A with Hugging Face")

repo_url = st.text_input("Enter GitHub Repository URL", "")

if repo_url:
    readme = fetch_github_readme(repo_url)
    if readme:
        st.subheader("README content:")
        st.write(readme[:2000] + "..." if len(readme) > 2000 else readme)
    else:
        st.error("Could not fetch README from the repository.")

    question = st.text_input("Ask a question about this repo:")

    if question:
        if st.button("Get Answer"):
            if not readme:
                st.error("No README content to answer from!")
            else:
                with st.spinner("Finding answer..."):
                    result = qa_pipeline(question=question, context=readme)
                    st.success(result["answer"])
