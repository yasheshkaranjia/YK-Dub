"""Quick review helper: prints Japanese / subtitle / final translation
side by side so you can eyeball translation quality before dubbing.
Usage: python check_translation.py <path to .translated.json>
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("Usage: python check_translation.py <path to .translated.json>")
    sys.exit(1)

with open(path, encoding="utf-8") as f:
    data = json.load(f)

for s in data["segments"]:
    print(f"[{s['flag']}] JP: {s['japanese_text']}")
    print(f"    SUB:   {s['subtitle_text']}")
    print(f"    FINAL: {s['final_text']}  (sim={s['similarity']})")
    print()
