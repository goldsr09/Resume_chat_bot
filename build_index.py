# build_index.py
import os, re, json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"

client = OpenAI()
DOC_PATH = "resume.txt"
INDEX_PATH = "index.faiss"
META_PATH = "meta.json"

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def chunk_text(txt, chunk_size=700, overlap=120):
    # paragraph-aware chunking, then add small overlap
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    final = []
    for i, c in enumerate(chunks):
        prev = chunks[i-1] if i > 0 else ""
        piece = (prev[-overlap:] + "\n\n" + c) if prev else c
        final.append(piece.strip())
    return final

def embed_texts(texts, batch=64):
    vecs = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_texts
        )
        vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    return np.vstack(vecs)  # (n, d)

def main():
    text = read_text(DOC_PATH)
    if not text:
        raise ValueError("resume.txt is empty — add your resume text first.")
    chunks = chunk_text(text)
    print(f"[build] chunks: {len(chunks)}")

    X = embed_texts(chunks)          # (n, d)
    faiss.normalize_L2(X)            # for cosine via inner product
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {INDEX_PATH} and {META_PATH}")

if __name__ == "__main__":
    main()
