"""
app.py
UGA Dorm Guide RAG Pipeline — Stage 3: Generation + Gradio Interface
---------------------------------------------------------------------
Pipeline position:
  fetch_raw.py        →  raw_docs/
  ingest_and_chunk.py →  chunks.json
  embed_and_store.py  →  chroma_db/        ← must exist before running this
  app.py              →  Gradio UI at http://localhost:7860

SETUP
  1. Get a free Groq API key at https://console.groq.com
  2. Create a .env file in your project folder with one line:
       GROQ_API_KEY=gsk_...your_key_here...
  3. Install dependencies:
       pip install groq gradio python-dotenv
  4. Make sure embed_and_store.py has already been run:
       python embed_and_store.py
  5. Launch:
       python app.py
     Then open http://localhost:7860 in your browser.

GROUNDING DESIGN (read this before changing the system prompt)
  Grounding means the LLM can ONLY answer from the retrieved chunks —
  it cannot use its own training knowledge about UGA dorms.

  This is enforced in TWO places, not one:
    1. The system prompt uses the word MUST and lists explicit refusal
       instructions — it does not just "suggest" grounding.
    2. Source attribution is built PROGRAMMATICALLY in format_response():
       the sources list is assembled from chunk metadata in Python, then
       appended to the LLM's answer. The LLM is never asked to produce
       the sources itself — it only produces the answer text.

  Why does this matter?
    If you only ask the LLM to "include sources", it may hallucinate URLs
    or titles it doesn't recognize. By building the source list in Python
    from the actual metadata of the retrieved chunks, attribution is always
    accurate — even if the LLM ignores the instructions.
"""

import os
from dotenv import load_dotenv
from groq import Groq
import gradio as gr

# Import retrieve() from the embedding/retrieval module
from embed_and_store import retrieve

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()   # reads GROQ_API_KEY from .env

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Create a .env file with: GROQ_API_KEY=gsk_..."
    )

# ── LLM config ────────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"   # free-tier, OpenAI-compatible
MAX_TOKENS   = 512
TEMPERATURE  = 0.2   # low = more factual, less creative

# ── System prompt ─────────────────────────────────────────────────────────────
# DESIGN NOTE: This prompt ENFORCES grounding — it does not merely suggest it.
#
# Key enforcement lines:
#   "You MUST answer using ONLY the information in the context passages below"
#   "Do NOT use any knowledge about UGA from your training data"
#   "If the context does not contain enough information ... say exactly:
#    'I don't have enough information in my sources to answer that.'"
#
# The word MUST and the exact refusal phrase make the constraint unambiguous.
# Vague instructions like "try to use the provided context" leave the LLM
# room to fill gaps with hallucinated information.
#
# What this prompt does NOT do: ask the LLM to list sources.
# Sources are appended programmatically by format_response() below.

SYSTEM_PROMPT = """You are a helpful advisor for University of Georgia (UGA) \
freshman choosing a dorm. You help students understand their housing options \
based on real student reviews, blog guides, and official UGA housing information.

You MUST answer using ONLY the information in the context passages provided \
with each question. Do NOT use any knowledge about UGA dorms, Athens, or \
campus life from your training data — even if you are confident about it.

Rules:
- Base every sentence of your answer on the provided context passages.
- Be specific: name dorms, quote prices, describe features when the context \
supports it.
- If the context contains conflicting opinions (common with Reddit sources), \
acknowledge both sides briefly.
- If the context does not contain enough information to answer the question, \
say exactly: "I don't have enough information in my sources to answer that."
- Do not make up dorm names, prices, locations, or features.
- Write in plain, friendly language — you are advising a freshman, not \
writing an academic paper.
- Do NOT include a list of sources or URLs in your answer. Sources are \
handled separately."""


# ── Context builder ───────────────────────────────────────────────────────────
def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.

    Each chunk is labelled with its source title and type so the LLM can
    refer to "according to the Reddit thread" or "the official UGA page says"
    naturally in its answer.

    Example output:
      [1] Source: The Ultimate Guide To Dorms At UGA (blog)
      Brumby Hall is a high-rise dorm known for its social atmosphere...

      [2] Source: Best dorms for freshman (reddit)
      [COMMENT] Brumby is amazing, my hallmates are my best friends...
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] Source: {chunk['source_title']} "
            f"({chunk['source_type']})"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


# ── Source list builder (PROGRAMMATIC — not LLM-generated) ───────────────────
def build_sources_md(chunks: list[dict]) -> str:
    """
    Build a deduplicated Markdown source list from chunk metadata.

    WHY THIS IS DONE IN PYTHON, NOT BY THE LLM:
      Asking the LLM to "list your sources" produces unreliable output —
      it may invent titles, garble URLs, or silently omit sources.
      By assembling the list from the actual metadata of retrieved chunks,
      we guarantee every source shown was genuinely used as context, with
      the correct title and URL.

    Deduplication: multiple chunks from the same URL are collapsed into
    one source entry so the list stays readable.
    """
    seen_urls = {}   # url → source_title (deduplication)
    for chunk in chunks:
        url   = chunk["url"]
        title = chunk["source_title"]
        stype = chunk["source_type"]
        if url not in seen_urls:
            seen_urls[url] = (title, stype)

    if not seen_urls:
        return ""

    lines = ["**Sources used:**"]
    for url, (title, stype) in seen_urls.items():
        lines.append(f"- [{title}]({url}) *({stype})*")
    return "\n".join(lines)


# ── Generation ────────────────────────────────────────────────────────────────
def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Send the query + retrieved context to Groq's LLaMA model and return
    the answer text (without sources — those are appended separately).

    The user message contains:
      - The numbered context passages (from build_context)
      - The question

    The system message enforces grounding (defined in SYSTEM_PROMPT above).
    """
    client  = Groq(api_key=GROQ_API_KEY)
    context = build_context(chunks)

    user_message = f"""Context passages:
{context}

Question: {query}

Answer based only on the context passages above:"""

    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Format final response ─────────────────────────────────────────────────────
def format_response(answer: str, chunks: list[dict]) -> str:
    """
    Combine the LLM answer with the programmatically-built source list.

    Output format:
      <answer text>

      ---
      **Sources used:**
      - [Title](url) *(type)*
      - ...
    """
    sources_md = build_sources_md(chunks)
    if sources_md:
        return f"{answer}\n\n---\n{sources_md}"
    return answer


# ── Main RAG function (called by Gradio) ─────────────────────────────────────
def ask_dorm_guide(query: str) -> str:
    """
    Full RAG pipeline for one query:
      1. Retrieve top-k chunks from ChromaDB
      2. Generate an answer with Groq LLaMA (grounded to context)
      3. Append programmatic source list
      4. Return formatted Markdown string to Gradio
    """
    query = query.strip()
    if not query:
        return "Please enter a question."

    # Step 1 — Retrieve
    chunks = retrieve(query)
    if not chunks:
        return "Could not retrieve any relevant information. Make sure embed_and_store.py has been run."

    # Step 2 — Generate
    answer = generate_answer(query, chunks)

    # Step 3 — Format with sources
    return format_response(answer, chunks)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
# STRUCTURE:
#   Header     — title + subtitle explaining what the tool does
#   Input      — single text box for the question
#   Output     — Markdown box (renders bold, links, bullet lists)
#   Examples   — the 5 eval questions from planning.md, pre-loaded as buttons
#   Footer     — disclaimer about source reliability

EXAMPLE_QUESTIONS = [
    "How do students view Russell Hall at UGA?",
    "What is the cheapest dorm at UGA?",
    "What are the benefits of a high rise dorm versus other dorms?",
    "What do students think of Brumby Hall?",
    "What dorms are coed at UGA?",
]

with gr.Blocks(title="UGA Dorm Guide") as demo:

    gr.Markdown("""
# 🐾 UGA Unofficial Dorm Guide
### AI-powered answers from real student reviews, blog guides, and official UGA housing data

Ask any question about UGA freshman dorms — which are social, which are quiet,
which are cheapest, or which have the best location. Answers are grounded in
the sources listed with each response.
    """)

    query_box  = gr.Textbox(
        label       = "Your question",
        placeholder = "e.g. What is the best dorm for introverts at UGA?",
        lines       = 2,
    )
    submit_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Markdown(label="Answer")

    # ── Example questions (one button each) ──────────────────────────────
    # Each button pre-fills the query box AND fires the RAG function.
    # Using gr.Examples is the simplest approach — it handles the wiring.
    gr.Examples(
        examples   = [[q] for q in EXAMPLE_QUESTIONS],
        inputs     = query_box,
        outputs    = answer_box,
        fn         = ask_dorm_guide,
        cache_examples = False,
        label      = "Try one of your 5 eval questions:",
    )

    # ── Wire submit button and Enter key ─────────────────────────────────
    submit_btn.click(
        fn      = ask_dorm_guide,
        inputs  = query_box,
        outputs = answer_box,
    )
    query_box.submit(
        fn      = ask_dorm_guide,
        inputs  = query_box,
        outputs = answer_box,
    )

    gr.Markdown("""
---
*Sources: student Reddit threads (r/UGA), blog guides, and official UGA Housing pages.
Opinions from Reddit reflect individual student experiences and may not represent
everyone's view. Always verify pricing and availability at
[housing.uga.edu](https://housing.uga.edu).*
    """)


if __name__ == "__main__":
    demo.launch()