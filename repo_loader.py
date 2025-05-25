import os
import requests
from urllib.parse import urlparse

def get_repo_info(url):
    """Extract owner and repo name from GitHub URL"""
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    else:
        raise ValueError("Invalid GitHub URL")

def fetch_readme(owner, repo):
    """Fetch the README.md file from the GitHub repo"""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print(f"README not found for {owner}/{repo}. Status: {response.status_code}")
        return None

if __name__ == "__main__":
    github_url = input("Enter GitHub repo URL: ").strip()

    try:
        owner, repo = get_repo_info(github_url)
        readme = fetch_readme(owner, repo)

        if readme:
            print("\n✅ README Content Preview (first 500 characters):\n")
            print(readme[:500])
        else:
            print("❌ Could not fetch README.")
        # Save README to file so we can use it in embedder.py
        with open("readme_cache.txt", "w", encoding="utf-8") as f:
            f.write(readme)
    except Exception as e:
        print(f"Error: {e}")
