import os, json
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import numpy as np
import faiss

load_dotenv()
client = OpenAI()  # uses OPENAI_API_KEY from .env

INDEX_PATH = "index.faiss"
META_PATH = "meta.json"

# Load index + chunks at startup
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise SystemExit("Missing index/meta. Run: python build_index.py")

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)["chunks"]

app = Flask(__name__)

def embed(query: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    v = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(v.reshape(1, -1))
    return v

def retrieve(query: str, k: int = 4):
    v = embed(query)
    D, I = index.search(v.reshape(1, -1), k)
    ctx = [chunks[i] for i in I[0]]
    return ctx, D[0].tolist()

def answer(query: str, context_snippets):
    system = (
        "You are a helpful assistant answering questions about Ryan Goldstein's resume. "
         "Use ONLY the provided context. If the answer isn't explicit, infer it from the context rather than saying 'I don't know.'"
    )
    context_block = "\n\n--- CONTEXT ---\n" + "\n\n---\n".join(context_snippets)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context_block}\n\nQuestion: {query}"},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Empty question"}), 400
    ctx, scores = retrieve(q, k=4)
    ans = answer(q, ctx)
    return jsonify({"answer": ans, "context": ctx, "scores": scores})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
