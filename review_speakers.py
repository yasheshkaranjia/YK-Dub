"""Review the Groq speaker labels before committing to any dubbing.

    python review_speakers.py "<...>.speakers.json"
"""
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
lines = data["lines"]

print(f"labelled {data['_labelled']}/{data['_total']} lines with {data['_model']}")
print()
print("=== FIRST 16 LINES - read these to judge if the labelling is sensible ===")
for row in lines[:16]:
    who = row["speaker"] or "?"
    print(f"  {row['start']:7.1f}s  [{who:>11}]  {row['text'][:60]}")

print()
print("=== A STRETCH OF 12 CONSECUTIVE LINES - a dialogue exchange ===")
mid = len(lines) // 2
for row in lines[mid:mid + 12]:
    who = row["speaker"] or "?"
    print(f"  {row['start']:7.1f}s  [{who:>11}]  {row['text'][:60]}")

print()
print("=== THE MAIN CHARACTER'S LINES (whatever Groq named most often) ===")
top = data["speakers"][0][0]
mine = [r for r in lines if r["speaker"] == top]
print(f'speaker "{top}": {len(mine)} lines')
for row in mine[:8]:
    print(f"  {row['start']:7.1f}s  {row['text'][:62]}")
print(f"  ... spans {mine[0]['start']:.0f}s to {mine[-1]['start']:.0f}s of the episode")

print()
print("=== UNLABELLED LINES (Groq could not decide) ===")
for row in [r for r in lines if not r["speaker"]][:8]:
    print(f"  {row['start']:7.1f}s  {row['text'][:62]}")