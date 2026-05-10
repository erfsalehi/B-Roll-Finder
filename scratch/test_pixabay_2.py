import os
import requests
from dotenv import load_dotenv

load_dotenv()

def search_pixabay(keyword, api_key, num_results=3):
    url = "https://pixabay.com/api/videos/"
    params = {"key": api_key, "q": keyword, "per_page": max(3, num_results)}
    print(f"Searching Pixabay for: '{keyword}'")
    response = requests.get(url, params=params)
    data = response.json()
    print(f"Total Hits: {data.get('totalHits', 0)}")
    for i, hit in enumerate(data.get('hits', [])):
        print(f"Hit {i}: {hit.get('tags')}")

api_key = os.getenv("PIXABAY_API_KEY")
search_pixabay("hand pulling out a screwdriver", api_key)
