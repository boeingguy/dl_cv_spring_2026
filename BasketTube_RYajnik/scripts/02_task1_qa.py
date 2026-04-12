import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat
import time

# ========================= CONFIG =========================
TRANSCRIPT_PATH = "data/processed/transcript.json"
LLM_MODEL = "qwen2.5:14b"
# =======================================================

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = json.load(f)

print(f"✅ Loaded {len(transcript)} segments\n")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [f"[{seg['start']:.2f}-{seg['end']:.2f}s] {seg['text']}" for seg in transcript]
embeddings = embedder.encode(texts, normalize_embeddings=True)

print("🎯 Task 1 QA System Ready!\n")

while True:
    query = input("💬 Your question: ").strip()
    if query.lower() in ["quit", "exit", "q"]:
        break
    if not query:
        continue

    start = time.time()

    # Retrieve top relevant segments
    query_emb = embedder.encode(query, normalize_embeddings=True)
    scores = np.dot(embeddings, query_emb)
    top_idx = scores.argsort()[-6:][::-1]        # top 6 best matches

    context = "\n".join([texts[i] for i in top_idx])

    system_prompt = """You are a professional basketball game analyst.
Answer using ONLY the provided commentary excerpts.
For every claim, cite the exact timestamp(s) like (123.45s).
Keep answers concise and factual.
Clearly separate commentator opinions from facts.
Always end with the exact sentence: 
"Note: This is based solely on commentary and may not reflect actual game events." """

    response = chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Commentary Excerpts:\n{context}\n\nQuestion: {query}"}
        ]
    )

    answer = response['message']['content']

    print("\n" + "═"*80)
    print(answer)
    print("═"*80)
    print(f"⏱️  {time.time()-start:.1f} seconds\n")
