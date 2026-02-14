import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

RAW_DOCS = "data/raw_docs"
STORE_PATH = "data/vector_store/store.json"

os.makedirs("data/vector_store", exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

store = []

for file in Path(RAW_DOCS).glob("**/*.txt"):
    text = file.read_text(encoding="utf-8")
    for c in chunk(text):
        store.append({
            "text": c,
            "embedding": model.encode(c).tolist()
        })

with open(STORE_PATH, "w", encoding="utf-8") as f:
    json.dump(store, f)

print("✅ Documents ingested")
