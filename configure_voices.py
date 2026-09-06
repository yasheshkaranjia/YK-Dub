"""
Interactive helper: scans a subtitle file for every speaking character
(sign/title lines are skipped automatically, same logic as script_agent.py)
and lets you assign each one to a Piper voice from voices.json. Saves your
choices to voice_map.json, which dub_agent.py reads at runtime.

Run it once per episode (or once per series if the same characters
recur) - it remembers earlier choices and only asks about new names it
hasn't seen before. Press Enter on any prompt to leave that character on
whatever it's currently set to (or the default voice, if never set).

Usage:
    python configure_voices.py <subtitle.ass>
"""
import json
import sys
from pathlib import Path

import pysubs2

from script_agent import is_sign_event

VOICES_FILE = Path(__file__).parent / "voices.json"
MAP_FILE = Path(__file__).parent / "voice_map.json"


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def find_speakers(sub_path: str):
    subs = pysubs2.load(sub_path)
    names = []
    for e in subs:
        if e.is_comment or not e.plaintext.strip() or is_sign_event(e):
            continue
        name = (e.name or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def run(sub_path: str) -> None:
    voices = load_json(VOICES_FILE, {})
    if not voices:
        print(f"No {VOICES_FILE.name} found, or it's empty - list your available "
              "Piper voices there first (model/config/label per voice).")
        return

    alias_list = list(voices.keys())
    voice_map = load_json(MAP_FILE, {"_default": alias_list[0]})
    speakers = find_speakers(sub_path)

    print("Available voices:")
    for i, alias in enumerate(alias_list, 1):
        print(f"  {i}. {alias} - {voices[alias].get('label', '')}")
    print(f"\nFound {len(speakers)} speaking characters. Press Enter to keep a "
          f"character's current/default voice, or type a number to change it.\n")

    for name in speakers:
        current = voice_map.get(name, f"(default: {voice_map['_default']})")
        choice = input(f"{name} [{current}]: ").strip()
        if not choice:
            continue
        try:
            idx = int(choice) - 1
            if idx < 0:
                raise ValueError
            voice_map[name] = alias_list[idx]
        except (ValueError, IndexError):
            print(f"  didn't understand '{choice}' - leaving {name} unchanged")

    MAP_FILE.write_text(json.dumps(voice_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {MAP_FILE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python configure_voices.py <subtitle.ass>")
        sys.exit(1)
    run(sys.argv[1])
