"""Tiny status readout for a long unattended YK-Dub run - prints one short
line per check so watching a 10-hour batch never floods the terminal.

    python dub_status.py "<work_root>" "<source folder>"

Shows: how many episodes are finished, which one is being worked on right
now and at which stage, plus the last few interesting log lines (the noisy
Demucs/tqdm progress bars are filtered out).
"""
import json
import sys
from pathlib import Path

VIDEO_EXTENSIONS = (".mkv", ".mp4")
INTERESTING = (
    "[extract]", "[script]", "[dub]", "[verify]", "[run]",
    "subtitle track", "WARNING", "tone-adjusted", "left it silent",
    "failed", "FAILED", "Done:", "===", "level 8", "burn",
)


def main() -> None:
    work_root = Path(sys.argv[1])
    source = Path(sys.argv[2])
    log = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(source.glob(f"*{ext}"))
    videos.sort()

    done, failed = [], []
    for video in videos:
        stem = video.stem
        if (work_root / stem / f"{stem}.dubbed.mp4").exists():
            done.append(stem)
        if (work_root / stem / f"{stem}.FAILED").exists():
            failed.append(stem)

    print(f"PROGRESS {len(done)}/{len(videos)} episodes finished")
    if done:
        print(f"  done:   {', '.join(d.split(' - ')[-1] for d in done)}")
    if failed:
        print(f"  FAILED: {', '.join(f.split(' - ')[-1] for f in failed)}")

    # Which stage is the current episode at? The heartbeat file the agents
    # write is the cheapest way to tell without parsing the log.
    hb = work_root / "heartbeat.json"
    if hb.exists():
        try:
            data = json.loads(hb.read_text(encoding="utf-8"))
            for stem, info in (data.items() if isinstance(data, dict) else []):
                if isinstance(info, dict) and info.get("stage"):
                    print(f"  current: {stem} -> {info['stage']}")
        except (OSError, ValueError):
            pass

    if log and log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [l.rstrip() for l in lines[-400:]
                if any(k in l for k in INTERESTING) and "|" not in l[:8]]
        print("  recent log:")
        for line in tail[-8:]:
            print(f"    {line[:150]}")


if __name__ == "__main__":
    main()