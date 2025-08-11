import os, json, hashlib, faiss, numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

def simple_token_estimate(s: str) -> int:
    return max(1, len(s) // 4)

def chunk_text(text: str, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    cur = []
    cur_tokens = 0
    for w in words:
        cur.append(w)
        cur_tokens += max(1, len(w)//4)
        if cur_tokens >= max_tokens:
            chunks.append(" ".join(cur))
            overlap_words = max(0, int(overlap*4))
            cur = cur[-overlap_words:] if overlap_words else []
            cur_tokens = sum(max(1, len(x)//4) for x in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

class SiteIndex:
    def __init__(self, site_id: str, base_dir: str = "data") -> None:
        self.site_id = site_id
        self.base_dir = base_dir
        self.dir = os.path.join(base_dir, site_id)
        os.makedirs(self.dir, exist_ok=True)
        self.model = SentenceTransformer(MODEL_NAME)
        self.index_path = os.path.join(self.dir, "faiss.index")
        self.docs_path = os.path.join(self.dir, "docs.jsonl")

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.docs_path)

    def _build_vectors(self, pages: List[Dict]) -> List[Dict]:
        docs = []
        for p in pages:
            chunks = chunk_text(p["text"]) or [p["text"]]
            for i, ch in enumerate(chunks):
                docs.append({
                    "id": hashlib.sha1(f"{p['url']}|{i}".encode("utf-8")).hexdigest(),
                    "url": p["url"],
                    "chunk_id": i,
                    "text": ch
                })
        return docs

    def build_or_update(self, pages: List[Dict]) -> int:
        docs = self._build_vectors(pages)
        with open(self.docs_path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        texts = [d["text"] for d in docs]
        embs = self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        embs = np.asarray(embs, dtype="float32")

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, self.index_path)
        return len(docs)

    def _load_index(self):
        index = faiss.read_index(self.index_path)
        return index

    def _iter_docs(self):
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def search(self, query: str, k: int = 6):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        index = self._load_index()
        D, I = index.search(q, k)
        docs = list(self._iter_docs())
        results = []
        for rank in range(len(I[0])):
            idx = int(I[0][rank])
            if idx < 0 or idx >= len(docs):
                continue
            results.append({
                "score": float(D[0][rank]),
                "doc": docs[idx]
            })
        return results
