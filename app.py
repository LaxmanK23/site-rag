import os, re, json, hashlib, asyncio, orjson
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crawler import crawl_site
from embedder import SiteIndex
from rag import answer_question

app = FastAPI(title="Website RAG Starter")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

class IngestReq(BaseModel):
    url: str

class AskReq(BaseModel):
    site_id: str
    question: str
    k: int = 6

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def home() -> HTMLResponse:
    html = open("static/index.html", "r", encoding="utf-8").read()
    return HTMLResponse(html)

@app.get("/sites")
def list_sites():
    if not os.path.isdir(DATA_DIR):
        return []
    items = []
    for sid in os.listdir(DATA_DIR):
        meta_path = os.path.join(DATA_DIR, sid, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            items.append(meta)
    return items

@app.post("/ingest")
async def ingest(req: IngestReq):
    site_id = hashlib.sha1(req.url.strip().encode("utf-8")).hexdigest()[:12]
    site_dir = os.path.join(DATA_DIR, site_id)
    os.makedirs(site_dir, exist_ok=True)

    pages = await crawl_site(req.url)
    if not pages:
        raise HTTPException(400, "No crawlable pages found (robots.txt or empty site).")

    indexer = SiteIndex(site_id=site_id, base_dir=DATA_DIR)
    n_chunks = indexer.build_or_update(pages)

    meta = {
        "site_id": site_id,
        "seed_url": req.url,
        "pages": len(pages),
        "chunks": n_chunks,
    }
    with open(os.path.join(site_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta

@app.post("/ask")
def ask(req: AskReq):
    indexer = SiteIndex(site_id=req.site_id, base_dir=DATA_DIR)
    if not indexer.exists():
        raise HTTPException(404, "Unknown site_id. Ingest first.")
    result = answer_question(req.question, indexer, k=req.k)
    return JSONResponse(result)
