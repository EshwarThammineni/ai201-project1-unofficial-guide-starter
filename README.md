# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

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

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:256**

**Overlap:50**

**Why these choices fit your documents: **
I first used 300 tokens but I found out that MiniLM truncates anything over 256 before embedding which means the last 44 tokens of each and every chunk were being dropped. Reducing it to 256 actually allows it to utilize the entire 256 tokens in every chunk. Additionally, a 50 token overlap was used in between chunks. The reason why we used 50  was because I found the sweet spot with trial and error to preserve the tail of one thought to the next without carrying much of the repeated context which boosts chunk count. 

**Final chunk count: 78 chunks across 9 sources**

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**
`all-MiniLM-L6-v2` was used. It was chosen because it was cheap and it was efficient for it to be able to run locally and embedd 78 chunks under 5 seconds. 

**Production tradeoff reflection:**
If cost wasn't an issue, I would utillize a stronger model. MiniLM does a great job at efficiency but it has trouble interpreting sarcastic/mixed reviews. 

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:** 
The system uses explicit instructions rather than mere suggestions. The instructions were: You MUST answer using ONLY the information in the context passages provided with each question. Do NOT use any knowledge about UGA dorms, Athens, or campus life from your training data — even if you are confident about it. If the context does not contain enough information to answer the question, say exactly: 'I don't have enough information in my sources to answer that.

**How source attribution is surfaced in the response:**
Source attribution is built in python after the LLM calls return. Because the sources come from the actual chunks rather than the LLM's output, every source shown was entirely used for context than from the LLM. 
---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 |How do students view Russell hall at UGA?|Russell hall is very social, has a nice location, and good study areas.|Its great for making friends and its regarded as "wild but fun"|100%|100%|
| 2 |What is the cheapest dorm option.|Brown Hall|The cheapesy dorm is Brown Hall|100%|100%|
| 3 |Whats the benefit of a high rise dorm versus other dorms.|High Rises are newer and more socical; Older dorms are larger but less social.|The high rise is newer,more social, and nicer.|100% |50%|
| 4 |What do students think of Brumby Hall|Students think positively of Brumby Hall as it is a quiet, relaxing place.|Students think that Brumby Hall is "quiet but beautiful" and "perfect if you’re not into the party scene."|100%|100%|
| 5 |What dorms isnt Coed|Soule Hall|Hill|100%|0%|

**Retrieval quality:** The retrieval quality is excellent. 
**Response accuracy:** Its partially acuurate. 

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** 
5

**What the system returned:** 
According to the context passages, Hill is the only dorm mentioned that is not coed, as it is described as "all girls".


**Root cause (tied to a specific pipeline stage):**
The likely issue is a chunk problem in the chunking stage where the information is being split. 

**What you would change to fix it:** You could simply just chabnge top k.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
This helped me the chunk was already there, it just wasn't making the cutoff. Increasing the k value would mean increasing the amount of chunks it accepts.  

**One way your implementation diverged from the spec, and why:** 
This is bad because now there are more chunks that the LLM processes and some might be irrelevent. 

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:*I gave claude my planning.md,strategy and told it to create ingestion and chunking code. 
- *What it produced:* It produced code that worked and produced chunks but only from certain sources. 
- *What I changed or overrode:* I overode the reddit sources manually because reddit shut down unauthenticated API access. So I manually copy and pasted the entire thread, saved it as a text file for the embedding model. 

**Instance 2**

- *What I gave the AI:* I gave Claude my injection, chunk code, and planning.md and told it to create a script for embedding. 
- *What it produced:* It produced the code to turn text into a list of numbers where similar meanings of text produce similar numbers. ChromaDB uses these numbers and finds the chunks whose question's nuumbers are compared with. 
- *What I changed or overrode:* The price question was wrong because the table is spread acorss 22 chunks. The K value was 6 which means it can only 6 the top 6 of those 22 chunks. Raising the K value means the llm can see more of the table and have more data. 
