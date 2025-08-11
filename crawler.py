import os, asyncio, urllib.parse, aiohttp, re
from typing import List, Dict, Tuple, Optional, Set
from bs4 import BeautifulSoup
import trafilatura
import urllib.robotparser as robotparser

MAX_PAGES = int(os.getenv("MAX_PAGES", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
CRAWL_DEPTH = int(os.getenv("CRAWL_DEPTH", "2"))

def same_host(a: str, b: str) -> bool:
    pa, pb = urllib.parse.urlparse(a), urllib.parse.urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)

def canonicalize(url: str) -> str:
    u = urllib.parse.urlparse(url)
    path = u.path or "/"
    path = re.sub(r"/+$", "/", path)
    new = urllib.parse.urlunparse((u.scheme, u.netloc, path, '', '', ''))
    return new

async def fetch_text(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200 or r.content_type not in ("text/html", "application/xhtml+xml"):
                return None
            return await r.text(errors="ignore")
    except Exception:
        return None

def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    hrefs = set()
    for a in soup.find_all('a', href=True):
        href = urllib.parse.urljoin(base_url, a['href'])
        if '#' in href:
            href = href.split('#', 1)[0]
        hrefs.add(href)
    return list(hrefs)

def clean_text(html: str, url: str) -> str:
    downloaded = trafilatura.extract(html, include_comments=False, include_tables=False, url=url)
    return downloaded or ""

async def crawl_site(seed_url: str) -> List[Dict]:
    seed = canonicalize(seed_url)
    parsed = urllib.parse.urlparse(seed)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass

    if rp.default_entry and not rp.can_fetch("*", seed):
        return []

    seen: Set[str] = set()
    q: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
    await q.put((seed, 0))

    results: List[Dict] = []
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(headers={"User-Agent": "SiteRAG/0.1"}) as session:
        while not q.empty() and len(results) < MAX_PAGES:
            url, depth = await q.get()
            url = canonicalize(url)
            if url in seen:
                continue
            seen.add(url)

            if not same_host(seed, url):
                continue
            if rp.default_entry and not rp.can_fetch("*", url):
                continue

            async with sem:
                html = await fetch_text(session, url)
            if not html:
                continue
            text = clean_text(html, url)
            if not text or len(text.strip()) < 200:
                continue

            results.append({"url": url, "text": text})

            if depth < CRAWL_DEPTH:
                for link in extract_links(html, url):
                    if same_host(seed, link):
                        await q.put((link, depth + 1))

            if len(results) >= MAX_PAGES:
                break

    uniq = {r["url"]: r for r in results}
    return list(uniq.values())
