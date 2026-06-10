"""
ingest_and_chunk.py
UGA Dorm Guide RAG Pipeline — Stage 1b: Cleaning & Chunking
------------------------------------------------------------
DEPENDS ON  fetch_raw.py  —  run that first to populate raw_docs/.

Two-stage design
  fetch_raw.py        →  raw_docs/src01_blog.txt … manifest.json
  ingest_and_chunk.py →  reads raw_docs/, cleans, chunks → chunks.json

Why split?
  Fetching hits the network; cleaning+chunking is pure CPU.
  Keeping them separate means you can tweak chunk size or clean logic
  and re-run in seconds without re-downloading anything.
"""

import re
import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from bs4 import BeautifulSoup

# load_raw_docs() reads manifest.json + .txt files from raw_docs/
from fetch_raw import load_raw_docs

# ── Tiktoken for accurate token counting ──────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
    def encode(text: str):
        return _enc.encode(text)
    def decode(tokens) -> str:
        return _enc.decode(tokens)
except ImportError:
    # Fallback: rough word-based approximation (1 word ≈ 1.3 tokens)
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)
    def encode(text: str):
        return text.split()
    def decode(tokens) -> str:
        return " ".join(tokens)


# ── Config: matches your planning.md spec exactly ─────────────────────────────
CHUNK_SIZE    = 256   # tokens — matches MiniLM max_seq_length exactly
CHUNK_OVERLAP = 50    # tokens




# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class RawDocument:
    source_id: int
    source_title: str
    source_type: str   # blog | reddit | official | review_site
    url: str
    raw_text: str


@dataclass
class Chunk:
    chunk_id: str          # e.g. "src1_chunk003"
    source_id: int
    source_title: str
    source_type: str
    url: str
    text: str
    token_count: int
    chunk_index: int


# ── Fetching is handled by fetch_raw.py ──────────────────────────────────────
# Run:  python fetch_raw.py
# That populates raw_docs/ which run_pipeline() reads below.


# ── Cleaning ──────────────────────────────────────────────────────────────────
# Noise patterns specific to your source types
_BLOG_NOISE = re.compile(
    r"(subscribe|newsletter|cookie|privacy policy|terms of use"
    r"|share this|follow us|pinterest|instagram|twitter|facebook"
    r"|advertisement|sponsored|all rights reserved"
    r"|©\s*\d{4})",
    re.IGNORECASE,
)

_WHITESPACE = re.compile(r"\n{3,}")   # collapse 3+ blank lines to 2


def clean_blog(html: str) -> str:
    """Extract article body from blog HTML, strip nav/footer noise."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove chrome
    for tag in soup(["nav", "header", "footer", "aside",
                     "script", "style", "noscript", "form",
                     "iframe", "figure"]):
        tag.decompose()

    # Prefer semantic article or main containers
    article = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"(post|article|content|entry)", re.I))
        or soup.body
    )
    text = article.get_text(separator="\n") if article else soup.get_text("\n")

    # Line-level noise filter
    lines = [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not _BLOG_NOISE.search(ln)
    ]
    cleaned = _WHITESPACE.sub("\n\n", "\n".join(lines))

    # ── Strip blog header lines (author, date, share counts, ToC) ──
    # These appear at the top of every article and add noise to chunk 0.
    header_noise = re.compile(
        r"^(by$|by\s|share|shares|\d+\s*shares?|\d+/\d+/\d+|table of contents"
        r"|understanding|top picks|considerations|final thoughts)",
        re.IGNORECASE,
    )
    cleaned_lines = [
        ln for ln in cleaned.splitlines()
        if not header_noise.match(ln.strip())
    ]
    cleaned = _WHITESPACE.sub("\n\n", "\n".join(cleaned_lines))
    
    return cleaned


def clean_official(html: str) -> str:
    """
    For UGA Housing pages, preserve table structure (rates are tabular).
    Convert <table> → plain-text rows so the chunker sees the data.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    parts = []
    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "table"]
    ):
        if element.name == "table":
            # Flatten table rows into pipe-separated lines
            for row in element.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in row.find_all(["th", "td"])]
                if any(cells):
                    parts.append(" | ".join(cells))
        else:
            txt = element.get_text(" ", strip=True)
            if txt and not _BLOG_NOISE.search(txt):
                parts.append(txt)

    return _WHITESPACE.sub("\n\n", "\n".join(parts))


def clean_review_site(html: str) -> str:
    """RateMyDorm: grab rating cards and review text blocks."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "iframe"]):
        tag.decompose()

    # Try to find review / rating blocks; fall back to full body text
    review_containers = soup.find_all(
        class_=re.compile(r"(review|rating|dorm|card|result)", re.I)
    )
    if review_containers:
        parts = []
        for c in review_containers:
            txt = c.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
        return _WHITESPACE.sub("\n\n", "\n".join(parts))
    return clean_blog(html)   # fallback


def clean_document(doc: RawDocument) -> str:
    """Dispatch to the right cleaner based on source type."""
    if doc.source_type == "reddit":
        text = doc.raw_text
        text = re.sub(r'\bUpvote\b.*?\bDownvote\b', '', text, flags=re.DOTALL)
        text = re.sub(r'\b(Reply|Award|Share)\b\n?', '', text)
        text = re.sub(r'u/\S+ avatar\n\S+\n', '', text)
        text = re.sub(r'•\n[\w\s]+ ago\n', '', text)
        text = re.sub(r'\d+ more repl\w+', '', text)
        return _WHITESPACE.sub("\n\n", text.strip())
    elif doc.source_type == "official":
        return clean_official(doc.raw_text)
    elif doc.source_type == "review_site":
        return clean_review_site(doc.raw_text)
    else:
        return clean_blog(doc.raw_text)   # blog is the default


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Token-aware sliding window chunker.

    How it works:
      1. Encode the entire cleaned text into token IDs.
      2. Slide a window of `chunk_size` tokens forward by
         (chunk_size - overlap) tokens each step.
      3. Decode each window back to a string.

    Why token-level (not character-level)?
      all-MiniLM-L6-v2 has a 256-token context window — going over that
      silently truncates your text and loses information. Counting tokens
      directly prevents that. 300-token chunks fit comfortably.

    Why 50-token overlap?
      Reddit comments and blog section headings sometimes land at chunk
      boundaries. A 50-token overlap (~2-3 sentences) ensures the heading
      or lead sentence of a thought is present in the next chunk too,
      so retrieval can surface it regardless of which chunk is fetched.
    """
    tokens = encode(text)
    if not tokens:
        return []

    step    = chunk_size - overlap
    chunks  = []
    start   = 0

    while start < len(tokens):
        end    = min(start + chunk_size, len(tokens))
        window = tokens[start:end]
        chunk_str = decode(window).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end == len(tokens):
            break
        start += step

    return chunks


# ── Pipeline orchestrator ─────────────────────────────────────────────────────
def run_pipeline(
    raw_dir:  str = "raw_docs",
    save_to:  str = "chunks.json",
) -> list[Chunk]:
    """
    Load raw files from raw_dir/, clean them, chunk them, save chunks.json.

    Step 0: load_raw_docs() reads manifest.json then each .txt file.
    Step 1: clean_document() strips noise (type-aware).
    Step 2: chunk_text() applies the 300-token / 50-token-overlap window.
    Step 3: write chunks.json with full metadata per chunk.
    """
    print(f"Loading raw documents from: {raw_dir}/")
    raw_docs = load_raw_docs(raw_dir)

    if not raw_docs:
        print("No documents loaded — run fetch_raw.py first.")
        return []

    all_chunks: list[Chunk] = []

    for entry in raw_docs:
        sid   = entry["source_id"]
        title = entry["source_title"]
        print(f"\n[{sid:02d}] Cleaning & chunking: {title}")

        doc = RawDocument(
            source_id    = sid,
            source_title = title,
            source_type  = entry["source_type"],
            url          = entry["url"],
            raw_text     = entry["raw_text"],
        )

        # 1. Clean
        cleaned = clean_document(doc)
        tok = count_tokens(cleaned)
        print(f"      → Cleaned: {tok:,} tokens")

        if tok < 50:
            print("      → Skipped (too little text after cleaning)")
            continue

        # 2. Chunk
        texts   = chunk_text(cleaned)
        skipped = 0
        for idx, text in enumerate(texts):
            tc = count_tokens(text)
            # Drop chunks that are too short to carry real content.
            # Blog header chunks (author name, date, share count) typically
            # land under 60 tokens and pollute retrieval results.
            if tc < 60:
                skipped += 1
                continue
            all_chunks.append(Chunk(
                chunk_id     = f"src{sid:02d}_chunk{idx:04d}",
                source_id    = sid,
                source_title = title,
                source_type  = entry["source_type"],
                url          = entry["url"],
                text         = text,
                token_count  = tc,
                chunk_index  = idx,
            ))
        kept = len(texts) - skipped
        print(f"      → Produced {kept} chunks"
              + (f" ({skipped} short chunks dropped)" if skipped else ""))

    # 3. Save
    out_path = Path(save_to)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(all_chunks)} chunks → {out_path}")
    return all_chunks


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks = run_pipeline(raw_dir="raw_docs", save_to="chunks.json")

    # Quick stats
    token_counts = [c.token_count for c in chunks]
    if token_counts:
        print(f"\n── Chunk Stats ───────────────────────")
        print(f"   Total chunks : {len(token_counts)}")
        print(f"   Avg tokens   : {sum(token_counts)/len(token_counts):.0f}")
        print(f"   Min tokens   : {min(token_counts)}")
        print(f"   Max tokens   : {max(token_counts)}")
        print(f"   (spec target : {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap)")