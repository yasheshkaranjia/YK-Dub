"""One-off pre-flight check: for every episode in a folder, list every
speaking character in its English subtitle and say whether voice_map.json
already has a voice for them. Read-only - touches nothing.

    python check_cast.py "<folder of episodes>" "<temp ass output>"
"""
import json
import subprocess
import sys
from pathlib import Path

import pysubs2

import script_agent

VIDEO_EXTENSIONS = (".mkv", ".mp4")


def main() -> None:
    folder = Path(sys.argv[1])
    scratch = Path(sys.argv[2])
    scratch.mkdir(parents=True, exist_ok=True)

    voice_map = json.loads(Path("voice_map.json").read_text(encoding="utf-8"))
    mapped = {k.upper() for k in voice_map if not k.startswith("_")}

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.glob(f"*{ext}"))
    videos.sort()

    unmapped_all = {}
    for video in videos:
        probe = scratch / f"{video.stem}.ass"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video),
             "-map", "0:s:0", "-c:s", "copy", str(probe)],
            check=True,
        )
        subs = pysubs2.load(str(probe))
        counts = {}
        for event in subs:
            if event.is_comment or not event.plaintext.strip():
                continue
            if script_agent.is_sign_event(event):
                continue
            name = (event.name or "").strip()
            counts[name] = counts.get(name, 0) + 1

        unmapped = {n: c for n, c in counts.items() if n and n.upper() not in mapped}
        flag = "OK" if not unmapped else "UNMAPPED"
        print(f"[{flag}] {video.name}: {sum(counts.values())} dialogue lines, "
              f"{len(counts)} speakers")
        for name, count in sorted(unmapped.items(), key=lambda kv: -kv[1]):
            print(f"         {count:4d}  {name!r}")
            unmapped_all[name] = unmapped_all.get(name, 0) + count

    if unmapped_all:
        print(f"\n{len(unmapped_all)} unmapped speaker(s) across the batch:")
        for name, count in sorted(unmapped_all.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {name!r}")
    else:
        print("\nEvery speaker in every episode has a voice assigned.")


if __name__ == "__main__":
    main()