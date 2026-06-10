"""
embed_and_store.py
UGA Dorm Guide RAG Pipeline — Stage 2: Embedding + Vector Store
---------------------------------------------------------------
Spec (from planning.md):
  Embedding model : all-MiniLM-L6-v2  (sentence-transformers)
  Vector store    : ChromaDB  (local, persistent)
  Top-k retrieval : k = 6

Pipeline position:
  fetch_raw.py          →  raw_docs/
  ingest_and_chunk.py   →  chunks.json          ← reads this
  embed_and_store.py    →  chroma_db/           ← writes this
  (Milestone 5)         →  generation / UI

USAGE
  # First time — embeds all chunks and saves to chroma_db/
  python embed_and_store.py

  # Query the store without re-embedding (already populated)
  python embed_and_store.py --query "What is the cheapest dorm at UGA?"

  # Force a full re-embed (if chunks.json changed)
  python embed_and_store.py --reset

WHAT THIS FILE CONTAINS
  load_chunks()       — reads chunks.json produced by ingest_and_chunk.py
  build_vector_store()— embeds every chunk with MiniLM and upserts into Chroma
  retrieve()          — embeds a query and returns the top-k most similar chunks
  main()              — CLI wiring
"""

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# ── Config — mirrors planning.md exactly ─────────────────────────────────────
CHUNKS_FILE     = "chunks.json"
CHROMA_DIR      = "chroma_db"          # folder Chroma persists its files to
COLLECTION_NAME = "uga_dorm_guide"     # logical name inside the Chroma database
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 6


# ── Step 1: Load chunks ───────────────────────────────────────────────────────
def load_chunks(path: str = CHUNKS_FILE) -> list[dict]:
    """
    Read chunks.json written by ingest_and_chunk.py.

    Each element looks like:
      {
        "chunk_id":     "src01_chunk0000",
        "source_id":    1,
        "source_title": "The Ultimate Guide To Dorms At UGA",
        "source_type":  "blog",
        "url":          "https://…",
        "text":         "…270–300 tokens of cleaned text…",
        "token_count":  298,
        "chunk_index":  0
      }
    """
    chunks_path = Path(path)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Run ingest_and_chunk.py first."
        )
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ── Step 2: Build the vector store ───────────────────────────────────────────
def build_vector_store(
    chunks:          list[dict],
    chroma_dir:      str  = CHROMA_DIR,
    collection_name: str  = COLLECTION_NAME,
    embed_model:     str  = EMBED_MODEL,
    reset:           bool = False,
) -> chromadb.Collection:
    """
    Embed every chunk with all-MiniLM-L6-v2 and upsert into ChromaDB.

    ── What is ChromaDB? ────────────────────────────────────────────────────
    ChromaDB is a vector database — a database designed to store and search
    by vector similarity rather than exact keyword match. Each row has:
      • an ID         (our chunk_id string)
      • a vector      (the 384-float embedding of the chunk text)
      • a document    (the raw text itself, stored for retrieval)
      • metadata      (source_title, source_type, url, token_count — anything
                       we want to filter on or display in results)

    ── What is PersistentClient? ────────────────────────────────────────────
    chromadb.PersistentClient(path="chroma_db") creates a local database
    stored in the chroma_db/ folder. It survives between runs — you only
    need to embed once; subsequent runs just query the existing store.
    (The alternative, chromadb.Client(), is in-memory and lost on exit.)

    ── What is get_or_create_collection? ────────────────────────────────────
    A Chroma database can hold multiple collections (like tables in SQL).
    get_or_create_collection() opens the named collection if it exists or
    creates it fresh if it doesn't. We tell it to use cosine distance
    (matching vectors pointing in the same direction = similar meaning),
    which is standard for sentence-transformer embeddings.

    ── What is upsert? ──────────────────────────────────────────────────────
    upsert = update + insert. If a chunk_id already exists in the collection
    it overwrites it; if it doesn't exist it inserts it. This means running
    the script twice won't create duplicate entries.

    ── Why embed in batches? ────────────────────────────────────────────────
    SentenceTransformer.encode() can take a list of strings and embeds them
    all at once (faster than one-by-one). We use batches of 64 to keep
    memory usage low — 65 chunks fit in 2 batches.
    """

    # ── Load the embedding model ──
    print(f"\nLoading embedding model: {embed_model}")
    print("  (downloads ~90 MB on first run, cached after that)")
    model = SentenceTransformer(embed_model)
    print(f"  Model max sequence length: {model.max_seq_length} tokens")

    # ── Connect to (or create) the local Chroma database ──
    print(f"\nConnecting to ChromaDB at: {chroma_dir}/")
    client = chromadb.PersistentClient(path=chroma_dir)

    # Optionally wipe the collection so we start fresh
    if reset:
        print(f"  --reset: deleting existing collection '{collection_name}'")
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass   # collection didn't exist yet — that's fine

    # get_or_create opens it if it exists, creates it if it doesn't
    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},
        # hnsw:space = "cosine" tells Chroma to use cosine similarity
        # (rather than Euclidean distance) when comparing vectors.
        # Cosine similarity measures the *angle* between two vectors —
        # chunks talking about the same topic point in the same direction
        # regardless of their length. This is the standard choice for
        # text embeddings from sentence-transformers.
    )

    existing = collection.count()
    if existing > 0 and not reset:
        print(f"  Collection already has {existing} vectors — skipping embed.")
        print(f"  Run with --reset to force a full re-embed.")
        return collection

    # ── Embed in batches and upsert ──
    BATCH = 64
    total  = len(chunks)
    print(f"\nEmbedding {total} chunks in batches of {BATCH}…")

    for start in range(0, total, BATCH):
        batch  = chunks[start : start + BATCH]
        end    = min(start + BATCH, total)

        texts  = [c["text"]       for c in batch]
        ids    = [c["chunk_id"]   for c in batch]

        # Metadata: anything you want to filter on or show in results.
        # ChromaDB requires all metadata values to be str, int, or float —
        # no nested dicts or lists.
        metadatas = [
            {
                "source_id":    c["source_id"],
                "source_title": c["source_title"],
                "source_type":  c["source_type"],
                "url":          c["url"],
                "token_count":  c["token_count"],
                "chunk_index":  c["chunk_index"],
            }
            for c in batch
        ]

        # encode() returns a numpy array of shape (batch_size, 384)
        # show_progress_bar gives a tqdm progress bar per batch
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        # .tolist() converts numpy float32 → plain Python floats,
        # which is what ChromaDB's upsert() expects.

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,      # store the raw text so we can read it back
            metadatas  = metadatas,
        )
        print(f"  Upserted chunks {start+1}–{end} / {total}")

    print(f"\n✓ Vector store ready — {collection.count()} vectors in '{collection_name}'")
    return collection


# ── Step 3: Retrieve ─────────────────────────────────────────────────────────
def retrieve(
    query:           str,
    collection:      chromadb.Collection = None,
    chroma_dir:      str  = CHROMA_DIR,
    collection_name: str  = COLLECTION_NAME,
    embed_model:     str  = EMBED_MODEL,
    k:               int  = TOP_K,
) -> list[dict]:
    """
    Hybrid retrieval: vector similarity + keyword boost for price queries.

    MiniLM doesn't reliably connect words like 'cheapest' or 'most expensive'
    to pipe-separated dollar amounts in the rates table. So after the normal
    vector search, if the query contains price-related keywords we also pull
    all src07 (rates) chunks and merge them in, de-duplicating by chunk_id.
    The merged list is re-ranked by distance so the best semantic matches
    still come first.
    """
    if collection is None:
        client     = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(collection_name)

    model           = SentenceTransformer(embed_model)
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = k,
        include          = ["documents", "metadatas", "distances"],
    )

    chunks_out = []
    seen_ids   = set()

    for doc, meta, dist, cid in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        chunks_out.append({
            "chunk_id":     cid,
            "text":         doc,
            "source_title": meta["source_title"],
            "source_type":  meta["source_type"],
            "url":          meta["url"],
            "distance":     round(dist, 4),
        })
        seen_ids.add(cid)

    # ── Keyword boost for price queries ──────────────────────────────────
    # If the query mentions cost/price/cheap/expensive, also fetch all
    # rates chunks (src07) and merge them in so the LLM has actual numbers.
    price_keywords = {"cheap", "cheapest", "expensive", "cost", "price",
                      "pricing", "affordable", "rates", "how much", "fee"}
    query_lower = query.lower()
    if any(kw in query_lower for kw in price_keywords):
        all_results = collection.get(
            include = ["documents", "metadatas"],
        )
        for doc, meta, cid in zip(
            all_results["documents"],
            all_results["metadatas"],
            all_results["ids"],
        ):
            if cid in seen_ids:
                continue
            if meta.get("source_id") == 7:   # src07 = UGA Housing Rates
                chunks_out.append({
                    "chunk_id":     cid,
                    "text":         doc,
                    "source_title": meta["source_title"],
                    "source_type":  meta["source_type"],
                    "url":          meta["url"],
                    "distance":     0.25,   # boost: treat as highly relevant
                })
                seen_ids.add(cid)

    # Re-rank by distance (lower = better) and return top k
    # For price queries, return enough chunks to cover the full rates table.
    # src07 has 22 chunks — we need all of them so the LLM can find the minimum.
    price_keywords = {"cheap", "cheapest", "expensive", "cost", "price",
                      "pricing", "affordable", "rates", "how much", "fee"}
    effective_k = 30 if any(kw in query.lower() for kw in price_keywords) else k
    chunks_out.sort(key=lambda x: x["distance"])
    return chunks_out[:effective_k]

# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks into ChromaDB and optionally run a test query."
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Test query to run after embedding (e.g. 'cheapest dorm at UGA')"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and re-embed the collection from scratch"
    )
    parser.add_argument(
        "--chunks", default=CHUNKS_FILE,
        help=f"Path to chunks.json (default: {CHUNKS_FILE})"
    )
    args = parser.parse_args()

    # 1. Load
    chunks = load_chunks(args.chunks)

    # 2. Embed + store
    collection = build_vector_store(chunks, reset=args.reset)

    # 3. Optional test query
    if args.query:
        print(f"\n── Query: \"{args.query}\" ──")
        print(f"   Retrieving top {TOP_K} chunks…\n")
        hits = retrieve(args.query, collection=collection)

        for i, hit in enumerate(hits, 1):
            print(f"  [{i}] {hit['chunk_id']}  distance={hit['distance']}")
            print(f"       Source : {hit['source_title']}  ({hit['source_type']})")
            print(f"       URL    : {hit['url']}")
            print(f"       Text   : {hit['text'][:200].strip()}…")
            print()


if __name__ == "__main__":
    main()