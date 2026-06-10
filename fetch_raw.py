"""
fetch_raw.py
UGA Dorm Guide RAG Pipeline — Stage 1a: Raw Document Fetching
--------------------------------------------------------------
PURPOSE
  Fetch every source and save the raw text to raw_docs/ BEFORE any
  cleaning or chunking happens.  This separation means:

    • You can re-run cleaning/chunking without hitting the network again.
    • You can open raw_docs/ and inspect what was actually captured.
    • If a site changes or goes down later, you still have the snapshot.
    • Each file is plain UTF-8 text, so you can diff, grep, or open in
      any editor.

OUTPUT LAYOUT
  raw_docs/
    manifest.json          ← index of every saved file + fetch metadata
    src01_blog.txt         ← "The Ultimate Guide To Dorms At UGA"
    src02_blog.txt         ← "The 5 Best University of Georgia Dorms"
    src03_blog.txt
    src04_review_site.txt
    src05_blog.txt
    src06_reddit.txt       ← Reddit thread: already extracted to plain text
    src07_official.txt     ← UGA Housing Rates (HTML preserved for table parsing)
    src08_reddit.txt
    src09_reddit.txt
    src10_official.txt     ← UGA Halls Information

FILE CONTENTS
  • Reddit sources  → plain text (post title + [COMMENT] blocks extracted
                       from the JSON API — no HTML to deal with downstream)
  • All other types → raw HTML  (the cleaning step in ingest_and_chunk.py
                       expects HTML for blogs, official pages, review sites)

USAGE
  python fetch_raw.py              # saves to ./raw_docs/
  python fetch_raw.py --out /tmp/raw_docs --skip-existing
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ── Source registry (mirrors planning.md Document Sources table) ──────────────
SOURCES = [
    {"id": 1,  "title": "The Ultimate Guide To Dorms At UGA",
     "type": "blog",
     "url": "https://www.society19.com/ultimate-guide-dorms-uga/"},

    {"id": 2,  "title": "The 5 Best University of Georgia Dorms",
     "type": "blog",
     "url": "https://humansofuniversity.com/university-of-georgia/the-5-best-university-of-georgia-dorm/"},

    {"id": 3,  "title": "Best University of Georgia Dorms: A Comprehensive Guide",
     "type": "blog",
     "url": "https://prked.com/post/best-university-of-georgia-dorms"},

    {"id": 4,  "title": "University of Georgia Freshman Dorms Ranked",
     "type": "review_site",
     "url": "https://www.ratemydorm.com/freshman-dorms-ranked/university-of-georgia"},

    {"id": 5,  "title": "Where to Live at the University of Georgia",
     "type": "blog",
     "url": "https://capgown.com/blogs/best-of/where-to-live-at-the-university-of-georgia-housing-options-for-the-bulldogs-community"},

    {"id": 6,  "title": "Best dorms for freshman (Reddit)",
     "type": "reddit",
     "url": "https://www.reddit.com/r/UGA/comments/17yf3k3/best_dorms_for_freshman/"},

    {"id": 7,  "title": "UGA Housing Rates",
     "type": "official",
     "url": "https://housing.uga.edu/rates/"},

    {"id": 8,  "title": "I was just admitted to UGA. What are the best dorms? (Reddit)",
     "type": "reddit",
     "url": "https://www.reddit.com/r/UGA/comments/1pfb5qc/i_was_just_admitted_to_uga_what_are_the_best_dorms/"},

    {"id": 9,  "title": "Dorm Rankings? (Reddit)",
     "type": "reddit",
     "url": "https://www.reddit.com/r/UGA/comments/1tx9qmr/dorm_rankings/"},

    {"id": 10, "title": "UGA Halls Information",
     "type": "official",
     "url": "https://housing.uga.edu/halls-information/"},
]


# Generic headers for blogs and official pages
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Reddit specifically requires a full browser-style User-Agent AND Accept headers.
# A bot-looking UA (like "UGA-DormGuide-RAG/1.0") returns 403 or a JSON error
# body even on the public .json endpoint. The Referer header also helps.
REDDIT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.reddit.com/",
}

# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class FetchResult:
    source_id:    int
    source_title: str
    source_type:  str        # blog | reddit | official | review_site
    url:          str
    filename:     str        # relative to raw_docs/  e.g. "src01_blog.txt"
    fetched_at:   str        # ISO-8601 UTC timestamp
    content_type: str        # "html" or "text"
    char_count:   int
    status:       str        # "ok" | "failed"
    error:        Optional[str] = None


# ── Fetchers ──────────────────────────────────────────────────────────────────
def fetch_html(url: str, retries: int = 2) -> Optional[str]:
    """
    Download a page and return the raw HTML string.
    Retries with exponential back-off on network errors.
    Returns None if all attempts fail.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt < retries:
                wait = 2 ** attempt   # 1s, 2s
                print(f"      [retry {attempt+1}] {exc}  — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"      [FAIL] {exc}")
                return None


def fetch_reddit(url: str, retries: int = 3) -> Optional[str]:
    """
    Fetch a Reddit thread via the public JSON API.

    WHY NOT SCRAPE HTML?
      Reddit's HTML page requires JavaScript to render — requests only gets
      a skeleton page or a 403. The .json endpoint is public and returns the
      full thread without any auth.

    WHY REDDIT_HEADERS (not the generic HEADERS)?
      Reddit checks the User-Agent server-side. The old custom UA string
      "UGA-DormGuide-RAG/1.0" was immediately identified as a bot and blocked
      with a 403 or a JSON response of {"error": 403}.
      A realistic Chrome UA + Accept + Referer passes that check.

    WHY THE DELAY BETWEEN RETRIES?
      Reddit rate-limits repeated requests from the same IP. If the first
      attempt gets a 429 (Too Many Requests), waiting a few seconds before
      retrying is usually enough to succeed.

    OUTPUT FORMAT
      [POST TITLE] …
      [POST BODY]  …   (omitted if the post has no text body)
      [COMMENT] …      (one block per non-deleted comment, incl. nested replies)
    """
    json_url = url.rstrip("/") + ".json?limit=200"

    for attempt in range(retries + 1):
        try:
            resp = requests.get(json_url, headers=REDDIT_HEADERS, timeout=20)

            # Catch rate-limit before raise_for_status so we can retry
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)   # 5s, 10s, 15s
                print(f"      [429 rate-limit] waiting {wait}s before retry…")
                time.sleep(wait)
                continue

            resp.raise_for_status()

            # Reddit sometimes returns 200 with a plain HTML error page
            # instead of JSON (e.g. if the thread was removed).
            # Detect that before trying to parse.
            if "application/json" not in resp.headers.get("Content-Type", ""):
                print(f"      [WARN] Reddit returned non-JSON content-type: "
                      f"{resp.headers.get('Content-Type')}")
                return None

            data = resp.json()

            # Sanity-check: a valid thread response is always a 2-element list
            if not isinstance(data, list) or len(data) < 2:
                print(f"      [WARN] Unexpected Reddit JSON shape: {str(data)[:120]}")
                return None

            break   # success — exit retry loop

        except requests.RequestException as exc:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"      [retry {attempt+1}] {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"      [FAIL] Reddit fetch gave up: {exc}")
                return None
    else:
        # All retries exhausted (only reached if every attempt hit 429)
        print(f"      [FAIL] Reddit fetch exhausted retries for {url}")
        return None

    # ── Extract post ──
    parts = []
    post = data[0]["data"]["children"][0]["data"]
    parts.append(f"[POST TITLE] {post.get('title', '').strip()}")
    body = post.get("selftext", "").strip()
    if body and body not in ("[deleted]", "[removed]"):
        parts.append(f"[POST BODY] {body}")

    # ── Extract comments (recursive — captures nested replies) ──
    def harvest(listing: dict) -> None:
        for child in listing.get("data", {}).get("children", []):
            cdata = child.get("data", {})
            text  = cdata.get("body", "").strip()
            if text and text not in ("[deleted]", "[removed]"):
                parts.append(f"[COMMENT] {text}")
            replies = cdata.get("replies")
            if isinstance(replies, dict):
                harvest(replies)

    harvest(data[1])
    return "\n\n".join(parts)


# ── Core save logic ───────────────────────────────────────────────────────────
def filename_for(src: dict) -> str:
    """
    Produce a stable, human-readable filename.
    e.g.  src01_blog.txt   src06_reddit.txt   src07_official.txt
    The zero-padded id means files sort in source order in any file browser.
    """
    return f"src{src['id']:02d}_{src['type']}.txt"


def fetch_and_save(
    src: dict,
    out_dir: Path,
    skip_existing: bool = False,
) -> FetchResult:
    """
    Fetch one source and write its raw content to out_dir/<filename>.
    Returns a FetchResult describing what happened (used to build the manifest).
    """
    fname   = filename_for(src)
    fpath   = out_dir / fname
    now_iso = datetime.now(timezone.utc).isoformat()

    # ── Skip if already on disk ──
    if skip_existing and fpath.exists():
        existing = fpath.read_text(encoding="utf-8")
        print(f"  [SKIP] {fname} already exists ({len(existing):,} chars)")
        return FetchResult(
            source_id    = src["id"],
            source_title = src["title"],
            source_type  = src["type"],
            url          = src["url"],
            filename     = fname,
            fetched_at   = now_iso,
            content_type = "text" if src["type"] == "reddit" else "html",
            char_count   = len(existing),
            status       = "ok (cached)",
        )

    # ── Fetch ──
    if src["type"] == "reddit":
        raw          = fetch_reddit(src["url"])
        content_type = "text"   # already plain text, no HTML
    else:
        raw          = fetch_html(src["url"])
        content_type = "html"   # blogs, official pages, review sites

    if raw is None:
        return FetchResult(
            source_id    = src["id"],
            source_title = src["title"],
            source_type  = src["type"],
            url          = src["url"],
            filename     = fname,
            fetched_at   = now_iso,
            content_type = content_type,
            char_count   = 0,
            status       = "failed",
            error        = "fetch returned None (see warnings above)",
        )

    # ── Write to disk ──
    fpath.write_text(raw, encoding="utf-8")

    print(f"  [OK]   {fname}  ({len(raw):,} chars, {content_type})")

    # Reddit rate-limits aggressively on consecutive requests — wait longer
    # between Reddit sources than between regular pages.
    return FetchResult(
        source_id    = src["id"],
        source_title = src["title"],
        source_type  = src["type"],
        url          = src["url"],
        filename     = fname,
        fetched_at   = now_iso,
        content_type = content_type,
        char_count   = len(raw),
        status       = "ok",
    )


# ── Manifest ──────────────────────────────────────────────────────────────────
def write_manifest(results: list[FetchResult], out_dir: Path) -> None:
    """
    Save a manifest.json that records every fetch result.
    This is the index used by ingest_and_chunk.py to load files
    without having to re-discover them.

    manifest.json format:
    {
      "fetched_at": "2025-…",
      "total": 10,
      "ok": 9,
      "failed": 1,
      "sources": [ { source_id, title, type, url, filename,
                      content_type, char_count, status, … }, … ]
    }
    """
    ok_count   = sum(1 for r in results if r.status.startswith("ok"))
    fail_count = sum(1 for r in results if r.status == "failed")

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total":      len(results),
        "ok":         ok_count,
        "failed":     fail_count,
        "sources":    [asdict(r) for r in results],
    }

    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"\n  Manifest → {path}")


# ── Loader (used by ingest_and_chunk.py) ──────────────────────────────────────
def load_raw_docs(raw_dir: str = "raw_docs") -> list[dict]:
    """
    Read every successfully-fetched file listed in manifest.json and
    return a list of dicts ready to construct RawDocument objects.

    Called by ingest_and_chunk.py so it doesn't need to know about
    filenames or the manifest structure.

    Returns list of:
      {
        "source_id":    int,
        "source_title": str,
        "source_type":  str,
        "url":          str,
        "content_type": str,   # "html" | "text"
        "raw_text":     str,
      }
    """
    raw_path     = Path(raw_dir)
    manifest_path = raw_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in {raw_dir}. "
            "Run fetch_raw.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = []

    for entry in manifest["sources"]:
        if not entry["status"].startswith("ok"):
            print(f"  [SKIP] src{entry['source_id']:02d} — status: {entry['status']}")
            continue

        fpath = raw_path / entry["filename"]
        if not fpath.exists():
            print(f"  [WARN] Listed in manifest but missing on disk: {fpath}")
            continue

        raw_text = fpath.read_text(encoding="utf-8")
        docs.append({
            "source_id":    entry["source_id"],
            "source_title": entry["source_title"],
            "source_type":  entry["source_type"],
            "url":          entry["url"],
            "content_type": entry["content_type"],
            "raw_text":     raw_text,
        })
        print(f"  [LOAD] src{entry['source_id']:02d}  {entry['filename']}"
              f"  ({len(raw_text):,} chars)")

    return docs


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all UGA dorm guide sources and save raw text to disk."
    )
    parser.add_argument(
        "--out", default="raw_docs",
        help="Output directory for raw files (default: ./raw_docs)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip sources whose file already exists in --out"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving raw documents to: {out_dir.resolve()}\n")

    results: list[FetchResult] = []

    for src in SOURCES:
        print(f"[{src['id']:02d}] {src['title']}")
        result = fetch_and_save(src, out_dir, skip_existing=args.skip_existing)
        results.append(result)
        if result.status == "ok":
            # Reddit rate-limits hard on back-to-back requests — use a longer
            # pause between Reddit threads than between regular pages.
            delay = 4 if src["type"] == "reddit" else 1
            time.sleep(delay)

    write_manifest(results, out_dir)

    # ── Summary ──
    ok   = sum(1 for r in results if r.status.startswith("ok"))
    fail = sum(1 for r in results if r.status == "failed")
    total_chars = sum(r.char_count for r in results)

    print(f"\n{'─'*44}")
    print(f"  Fetched : {ok}/{len(results)} sources")
    if fail:
        failed_titles = [r.source_title for r in results if r.status == "failed"]
        for t in failed_titles:
            print(f"  FAILED  : {t}")
    print(f"  Total   : {total_chars:,} characters saved")
    print(f"  Output  : {out_dir.resolve()}/")
    print(f"{'─'*44}")

    if fail:
        sys.exit(1)   # non-zero exit so CI/make can detect partial failure


if __name__ == "__main__":
    main()