# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
This domain focuses on UGA’s on campus dorm reviews, and helps freshman choose the dorm that is right for them. There are a lot of options and all of them have their ups and downs, which makes this guide essential for summarizing, whats the go to, and what to stick away from. 

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 |The Ultimate Guide To Dorms At UGA|Blog|https://www.society19.com/ultimate-guide-dorms-uga/ |
| 2 |The 5 Best University of Georgia Dorms|Blog| |https://humansofuniversity.com/university-of-georgia/the-5-best-university-of-georgia-dorm/
| 3 |Best University of Georgia Dorms: A Comprehensive Guide|Blog|https://prked.com/post/best-university-of-georgia-dorms|
| 4 |University of Georgia Freshman Dorms Ranked|Reviews|https://www.ratemydorm.com/freshman-dorms-ranked/university-of-georgia|
| 5 |Where to Live at the University of Georgia: Housing Options for The Bulldogs Community|Blog|https://capgown.com/blogs/best-of/where-to-live-at-the-university-of-georgia-housing-options-for-the-bulldogs-community|
| 6 |Best dorms for freshman|Forum|https://www.reddit.com/r/UGA/comments/17yf3k3/best_dorms_for_freshman/|
| 7 |Rates|Official University Source||https://housing.uga.edu/rates/|
| 8 |I was just admitted to UGA. What are the best dorms?|Forum|https://www.reddit.com/r/UGA/comments/1pfb5qc/i_was_just_admitted_to_uga_what_are_the_best_dorms/|
| 9 |Dorm Rankings?|Forum|https://www.reddit.com/r/UGA/comments/1tx9qmr/dorm_rankings/|
| 10 |Halls Information|Official University Source|https://housing.uga.edu/halls-information/|

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:300 tokens**  

**Overlap: 50 tokens**

**Reasoning: The sources comprise of reddit posts which are conversational, Blog style guides which sections are structured by dorm, and official sources from the University which are strucutured tables.A medium chunk size allows the description or reddit thread to be fully contained. The small overlap size allows for context making it more effective.**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:all-MiniLM-L6-v2**

**Top-k:6**

**Production tradeoff reflection:Models like the MiniLM are fast and cheap but they might misinterprit neutral/sarcastic reviews. Additionally reddit threads often include slang which makes it difficulty for the model to guage sentiment.**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 |How do students view Russell hall at UGA?|Russell hall is very social, has a nice location, and good study areas.|
| 2 |What is the cheapest dorm option.|Brown Hall|
| 3 |Whats the benefit of a high rise dorm versus other dorms.|High Rises are newer and more socical; Older dorms are larger but less social.|
| 4 |What do students think of Brumby Hall|Students think positively of Brumby Hall as it is a quiet, relaxing place|
| 5 |What dorms isnt Coed|Soule Hall.|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. There could be conflicing opinions across the sources.

2. Many sources say top 5 dorms which could add bias to the results. 

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

     1:Document Ingestion
          (Python)
            
     2:Chunking
          (300 tokens, 50 overlap)
            
     3:Embedding + Vector Store
          (all-MiniLM-L6-v2 + ChromaDB)
            
     4:Retrieval
          (top-k, k=6)
            
     5:Generation
          (Grok API)
---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
