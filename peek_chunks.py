import json
import random

chunks = json.load(open("chunks.json", encoding="utf-8"))
sample = random.sample(chunks, 5)

for c in sample:
    print(f"\n{'─'*60}")
    print(f"ID     : {c['chunk_id']}")
    print(f"Source : {c['source_title']}")
    print(f"Type   : {c['source_type']}")
    print(f"Tokens : {c['token_count']}")
    print(f"{'─'*60}")
    print(c["text"])

print(f"\n{'─'*60}")
print(f"Total chunks in file: {len(chunks)}")