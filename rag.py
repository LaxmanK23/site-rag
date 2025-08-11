import os
from typing import Dict, Any, List
from embedder import SiteIndex
from openai import OpenAI

SYSTEM_PROMPT = """You are a helpful assistant. Answer based only on the provided context.
If the answer isn't in the context, say you don't know and suggest where to look on the site.
Cite sources as [n] with their URLs."""

def format_context(hits: List[Dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        url = h["doc"]["url"]
        txt = h["doc"]["text"]
        blocks.append(f"[Source {i}] {url}\n{txt}")
    return "\n\n".join(blocks)

def build_extractive_answer(hits: List[Dict], question: str) -> Dict[str, Any]:
    top = hits[:3]
    answer = "\n\n".join(h["doc"]["text"][:700] for h in top)
    sources = [{"url": h["doc"]["url"], "score": h["score"]} for h in top]
    return {"answer": answer, "sources": sources, "mode": "extractive"}

def build_llm_answer(hits: List[Dict], question: str) -> Dict[str, Any]:
    client = OpenAI()
    content = f"Question: {question}\n\nContext:\n{format_context(hits)}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": content},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    sources = [{"url": h["doc"]["url"], "score": h["score"]} for h in hits[:5]]
    return {"answer": text, "sources": sources, "mode": "llm"}

def answer_question(question: str, index: SiteIndex, k: int = 6):
    hits = index.search(question, k=k)
    if not hits:
        return {"answer": "I couldn't retrieve anything for that site yet.", "sources": [], "mode": "none"}
    if os.getenv("OPENAI_API_KEY"):
        return build_llm_answer(hits, question)
    else:
        return build_extractive_answer(hits, question)
