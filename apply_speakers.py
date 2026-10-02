"""Write Groq's speaker labels back into a .translated.json.

    python apply_speakers.py "<...>.translated.json" [--unlabelled NAME]

WHY THIS IS NEEDED
script_agent.py records each line's speaker from the ASS "Name" field.
This release leaves that field empty on all 400 lines, so every line lands
as one unnamed speaker and voice_map.json has nothing to match. Groq's
labels (speaker_id_groq.py) fill that gap - this copies them in, and does it
NON-DESTRUCTIVELY: a .bak of the original JSON is written first, and any
line that already had a real name in the subtitle is left alone, since a
fansub's own tagging is more authoritative than an inferred one.

--unlabelled decides what happens to lines Groq could not decide. Default
is to leave them empty, which routes them to the _default voice. Passing a
name (e.g. --unlabelled <CharacterName>) assigns them to that character instead,
which is usually better than a character nobody has heard.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("translated_json")
    parser.add_argument("--unlabelled", default=None,
                        help="speaker name for lines Groq left blank")
    args = parser.parse_args()

    src = Path(args.translated_json)
    speakers_file = src.with_name(src.name.replace(".translated.json", ".speakers.json"))
    if not speakers_file.exists():
        raise SystemExit(f"missing {speakers_file.name} - run speaker_id_groq.py first")

    backup = src.with_suffix(".prelabels.bak.json")
    shutil.copy(src, backup)

    data = json.loads(src.read_text(encoding="utf-8"))
    segments = data["segments"]
    labels = json.loads(speakers_file.read_text(encoding="utf-8"))["lines"]

    if len(labels) != len(segments):
        raise SystemExit(f"label count {len(labels)} != segment count "
                         f"{len(segments)} - the two files are out of step")

    filled = blanked = unchanged = 0
    for seg, row in zip(segments, labels):
        existing = (seg.get("speaker") or "").strip()
        inferred = (row.get("speaker") or "").strip()
        if existing:
            # A name the subtitle itself carried wins - do not overwrite it.
            unchanged += 1
            continue
        if inferred:
            seg["speaker"] = inferred
            filled += 1
        elif args.unlabelled:
            seg["speaker"] = args.unlabelled
            filled += 1
        else:
            blanked += 1

    src.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = Counter((s.get("speaker") or "(blank)").strip() for s in segments)
    print(f"[apply] {src.name}")
    print(f"[apply]   filled from Groq : {filled}")
    print(f"[apply]   left as-is (had a name already) : {unchanged}")
    print(f"[apply]   still blank      : {blanked}")
    print(f"[apply]   backup: {backup.name}")
    print()
    print(f"{'lines':>6}  speaker")
    print("-" * 40)
    for name, count in counts.most_common():
        print(f"{count:6d}  {name}")


if __name__ == "__main__":
    main()