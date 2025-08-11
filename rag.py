import os, requests
from typing import Dict, Any, List
from embedder import SiteIndex
from sentence_transformers import CrossEncoder

SYSTEM_PROMPT = """You are a helpful assistant. Answer based only on the provided context.
If the answer isn't in the context, say you don't know and suggest where to look on the site.
Cite sources as [n] with their URLs."""

# Lazy global for reranker
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        # small, fast cross-encoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def format_context(hits: List[Dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        url = h["doc"]["url"]
        txt = h["doc"]["text"]
        blocks.append(f"[Source {i}] {url}\n{txt}")
    return "\n\n".join(blocks)

def maybe_rerank(hits: List[Dict], question: str, use_rerank: bool) -> List[Dict]:
    if not use_rerank or len(hits) <= 2:
        return hits
    ce = get_reranker()
    pairs = [(question, h["doc"]["text"]) for h in hits]
    scores = ce.predict(pairs).tolist()
    scored = [{"score": float(s), "doc": h["doc"]} for s, h in zip(scores, hits)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def build_extractive_answer(hits: List[Dict], question: str) -> Dict[str, Any]:
    top = hits[:3]
    answer = "\n\n".join(h["doc"]["text"][:700] for h in top)
    sources = [{"url": h["doc"]["url"], "score": h.get("score", 0.0)} for h in top]
    return {"answer": answer, "sources": sources, "mode": "extractive"}

def build_openai_answer(hits: List[Dict], question: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    from openai import OpenAI
    m = model or "gpt-4o-mini"
    content = f"Question: {question}\n\nContext:\n{format_context(hits)}"
    client = OpenAI()
    resp = client.chat.completions.create(
        model=m,
        messages=[{"role":"system","content": SYSTEM_PROMPT},
                  {"role":"user","content": content}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    text = resp.choices[0].message.content
    sources = [{"url": h["doc"]["url"], "score": h.get("score", 0.0)} for h in hits[:5]]
    return {"answer": text, "sources": sources, "mode": f"openai:{m}"}

def build_ollama_answer(hits: List[Dict], question: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    m = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
    content = f"Question: {question}\n\nContext:\n{format_context(hits)}"
    payload = {
        "model": m,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(max_tokens)}
    }
    r = requests.post(f"{host}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    text = (data.get("message") or {}).get("content", "") or data.get("response", "")
    sources = [{"url": h["doc"]["url"], "score": h.get("score", 0.0)} for h in hits[:5]]
    return {"answer": text, "sources": sources, "mode": f"ollama:{m}"}

def answer_question(
    question: str,
    index: SiteIndex,
    k: int = 6,
    gen_mode: str = "",
    model: str = "",
    temperature: float = 0.2,
    max_tokens: int = 512,
    rerank: bool = False,
):
    hits = index.search(question, k=k)
    if not hits:
        return {"answer": "I couldn't retrieve anything for that site yet.", "sources": [], "mode": "none"}

    # optional reranking
    hits = maybe_rerank(hits, question, rerank)

    gm = (gen_mode or "").lower()
    if gm == "extractive" or gm == "":
        # If unspecified, fall back to OpenAI if key present; else extractive
        if gm == "" and os.getenv("OPENAI_API_KEY"):
            try:
                return build_openai_answer(hits, question, model, temperature, max_tokens)
            except Exception:
                pass
        return build_extractive_answer(hits, question)

    if gm == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return {"answer": "OPENAI_API_KEY not set. Switch to Extractive or Ollama.", "sources": [], "mode": "error"}
        try:
            return build_openai_answer(hits, question, model, temperature, max_tokens)
        except Exception as e:
            return {"answer": f"OpenAI error: {e}", "sources": [], "mode": "error"}

    if gm == "ollama":
        try:
            return build_ollama_answer(hits, question, model, temperature, max_tokens)
        except Exception as e:
            return {"answer": f"Ollama error: {e}", "sources": [], "mode": "error"}

    return build_extractive_answer(hits, question)
