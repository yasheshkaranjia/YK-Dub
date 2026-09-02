"""
Quick review helper: prints each dialogue line's timing, speaker, and
final text before you commit to a full dub run - useful for catching a
speaker name that didn't get picked up, or a sign line that slipped
through as dialogue by mistake.

Usage: python check_translation.py <path to .translated.json>
Add --speaker "NAME" to show only one character's lines.
"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python check_translation.py <path to .translated.json> [--speaker NAME]")
    sys.exit(1)

path = sys.argv[1]
speaker_filter = None
if "--speaker" in sys.argv:
    idx = sys.argv.index("--speaker")
    speaker_filter = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

with open(path, encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]
if speaker_filter:
    segments = [s for s in segments if s.get("speaker", "") == speaker_filter]

for s in segments:
    speaker = s.get("speaker") or "(unnamed)"
    duration = s["end"] - s["start"]
    print(f"[{s['start']:7.2f}s -> {s['end']:7.2f}s | {duration:4.1f}s | {speaker}]")
    print(f"    {s['final_text']}")
    print()

print(f"{len(segments)} line(s) shown"
      f"{f' for speaker \"{speaker_filter}\"' if speaker_filter else ''} "
      f"out of {len(data['segments'])} total.")
if data.get("signs_path"):
    print(f"Sign/overlay text (not shown here, burned onto video separately): {data['signs_path']}")
