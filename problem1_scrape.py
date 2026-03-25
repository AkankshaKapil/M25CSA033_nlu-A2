"""
Problem 1 - Task 1: Web CRAWLER for IIT Jodhpur Data
=====================================================
Instead of a fixed URL list, this crawler:
  - Starts from seed URLs (homepage + key department pages)
  - Automatically discovers and follows ALL links within iitj.ac.in
  - Visits up to MAX_PAGES pages (set to 300 for a rich corpus)
  - Skips PDFs, images, login pages, and non-English content
  - Saves each page as a separate document + merged corpus.txt

Expected output:
  - 150-300 documents
  - 100,000-500,000 tokens
  - Much richer vocabulary for Word2Vec

Usage:
    python problem1_scrape.py

Time: ~5-15 minutes depending on network speed and MAX_PAGES
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import re, time, os, json, random

# Configuration
MAX_PAGES    = 300
DELAY_MIN    = 0.5
DELAY_MAX    = 1.2
TIMEOUT      = 15
MIN_WORDS    = 40
OUT_DIR      = "data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# Allowed domains - crawler stays within these
ALLOWED_DOMAINS = {
    "www.iitj.ac.in",
    "iitj.ac.in",
    "old.iitj.ac.in",
    "academics.iitj.ac.in",
}

# Seed URLs - crawler starts here and discovers everything else
SEED_URLS = [
    "https://www.iitj.ac.in/",
    "https://www.iitj.ac.in/main/en/about-iit-jodhpur",
    "https://www.iitj.ac.in/office-of-academics/en/academic-programs",
    "https://www.iitj.ac.in/office-of-academics/en/circulars",
    "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Calendar",
    "https://www.iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://www.iitj.ac.in/office-of-academics/en/curriculum",
    "https://www.iitj.ac.in/computer-science-engineering/en/faculty",
    "https://www.iitj.ac.in/computer-science-engineering/en/research",
    "https://www.iitj.ac.in/electrical-engineering/en/faculty-members",
    "https://www.iitj.ac.in/mechanical-engineering/en/faculty-members",
    "https://www.iitj.ac.in/mathematics/en/faculty-members",
    "https://www.iitj.ac.in/physics/en/faculty-members",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/faculty-members",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/research",
    "https://www.iitj.ac.in/research-and-development/en/overview",
    "https://www.iitj.ac.in/students/en/student-life",
    "https://www.iitj.ac.in/placements-training-internships/en/placements",
    "https://www.iitj.ac.in/faculty-positions/en/faculty-positions",
    "http://academics.iitj.ac.in/",
]

# File extensions to skip
SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".zip", ".rar", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".avi", ".mov",
    ".css", ".js", ".xml", ".json",
}

# URL patterns to skip
SKIP_PATTERNS = [
    r"/login", r"/logout", r"/signin", r"/signup",
    r"/wp-admin", r"/admin",
    r"mailto:", r"tel:", r"javascript:",
    r"\?.*page=\d{3,}",
    r"/feed/", r"/rss",
    r"intranet",
]


def should_skip_url(url):
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc not in ALLOWED_DOMAINS:
        return True
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True
    if any(re.search(pat, url, re.IGNORECASE) for pat in SKIP_PATTERNS):
        return True
    return False


def clean_text(raw):
    # Strip non-ASCII (removes Devanagari/Hindi)
    text = raw.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\;\:\-\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_and_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")

    # Collect links before removing tags
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        full_url = urljoin(base_url, href).split("#")[0]
        if full_url and not should_skip_url(full_url):
            links.append(full_url)

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "button",
                     "meta", "link", "figure", "figcaption"]):
        tag.decompose()

    raw_text = soup.get_text(separator=" ")
    return clean_text(raw_text), links


def crawl(seed_urls, max_pages=MAX_PAGES, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    queue   = deque(seed_urls)
    visited = set()
    docs    = []
    skipped = 0

    print("="*60)
    print("  IIT JODHPUR WEB CRAWLER")
    print(f"  Max pages : {max_pages}")
    print(f"  Seeds     : {len(seed_urls)}")
    print("="*60)

    while queue and len(docs) < max_pages:
        url = queue.popleft().rstrip("/")
        if not url or url in visited:
            continue
        visited.add(url)

        print(f"[{len(docs)+1:03d}/{max_pages}] {url[:75]}")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                                allow_redirects=True)
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                print(f"         [SKIP] Not HTML")
                skipped += 1
                continue
            resp.raise_for_status()
        except Exception as e:
            print(f"         [FAIL] {str(e)[:55]}")
            skipped += 1
            continue

        text, new_links = extract_text_and_links(resp.text, url)
        word_count = len(text.split())

        if word_count < MIN_WORDS:
            print(f"         [SKIP] Only {word_count} words")
            skipped += 1
        else:
            docs.append({"url": url, "text": text, "words": word_count})
            print(f"         OK  {word_count:,} words | docs so far: {len(docs)}")

        # Add newly discovered links to queue
        for link in new_links:
            if link not in visited:
                queue.append(link)

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # Save raw JSON
    with open(os.path.join(out_dir, "raw_docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=True)

    # Save merged corpus
    corpus = "\n\n".join(d["text"] for d in docs)
    with open(os.path.join(out_dir, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write(corpus)

    total_words = sum(d["words"] for d in docs)
    print("\n" + "="*60)
    print(f"  CRAWL COMPLETE")
    print(f"  Pages saved    : {len(docs)}")
    print(f"  Pages skipped  : {skipped}")
    print(f"  Total words    : {total_words:,}")
    print(f"  Corpus size    : {len(corpus):,} chars")
    print(f"  Saved to       : {out_dir}/corpus.txt")
    print("="*60)
    return docs


if __name__ == "__main__":
    crawl(SEED_URLS, max_pages=MAX_PAGES)