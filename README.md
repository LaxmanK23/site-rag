# Website RAG Starter

Paste a site URL → crawl permitted pages → extract clean text → chunk + embed → ask questions (RAG).

## Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```
