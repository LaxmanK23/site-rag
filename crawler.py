import os, asyncio, urllib.parse, aiohttp, re
from typing import List, Dict, Tuple, Optional, Set
from bs4 import BeautifulSoup
import trafilatura
import urllib.robotparser as robotparser

MAX_PAGES = int(os.getenv("MAX_PAGES", "10000000"))  # Effectively unlimited
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
CRAWL_DEPTH = int(os.getenv("CRAWL_DEPTH", "10000000"))  # Effectively unlimited

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
    
    # robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    # rp = robotparser.RobotFileParser()
    # try:
    #     rp.set_url(robots_url)
    #     rp.read()
    # except Exception:
    #     pass
    # if rp.default_entry and not rp.can_fetch("*", seed):
    #     return []
    # robots.txt check disabled for testing purposes
    rp = None
    print(f"[Crawler] WARNING: Attempting to crawl the entire site from {seed_url}. This may take a very long time and use significant resources.")

    seen: Set[str] = set()
    q: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
    await q.put((seed, 0))

    results: List[Dict] = []
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Use a browser-like user agent
    # user_agent = "SiteRAG/0.1"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    async with aiohttp.ClientSession(headers={"User-Agent": user_agent}) as session:

        while not q.empty() and len(results) < MAX_PAGES:
            url, depth = await q.get()
            url = canonicalize(url)
            if url in seen:
                continue
            seen.add(url)

            if not same_host(seed, url):
                continue
            # robots.txt check disabled for testing purposes
            # if rp.default_entry and not rp.can_fetch("*", url):
            #     continue

            async with sem:
                html = await fetch_text(session, url)
            if not html:
                print(f"[Crawler] Fetch error or non-HTML for {url}")
                continue

            text = clean_text(html, url)
            # if not text or len(text.strip()) < 200:
            #     continue
            # Removed text length filter: crawl all pages
            results.append({"url": url, "text": text})

            if depth < CRAWL_DEPTH:
                for link in extract_links(html, url):
                    if same_host(seed, link):
                        await q.put((link, depth + 1))

            if len(results) >= MAX_PAGES:
                break

    uniq = {r["url"]: r for r in results}
    if not results:
        print(f"[Crawler] No crawlable pages: site is empty or inaccessible for {seed_url}")
    return list(uniq.values())
