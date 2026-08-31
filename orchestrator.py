"""
Orchestrator: runs the 4-agent pipeline end to end for one video, or for
every video in a folder - one episode at a time, which is kinder to a
low-end laptop than trying to run several in parallel. Queue a whole
season overnight and check results in the morning.
"""
import sys
from pathlib import Path

import extract_agent
import translate_agent
import dub_agent
import verify_agent


def process_episode(video_path: str, work_root: str) -> None:
    stem = Path(video_path).stem
    work_dir = Path(work_root) / stem
    print(f"\n=== {stem} ===")

    extract_agent.run(video_path, str(work_dir))
    manifest_path = work_dir / f"{stem}.manifest.json"

    translate_agent.run(str(manifest_path))
    translated_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))

    out_video = work_dir / f"{stem}.dubbed.mp4"
    dub_agent.run(str(translated_path), str(out_video))
    dubbed_path = Path(str(translated_path).replace(".translated.json", ".dubbed.json"))

    verify_agent.run(str(dubbed_path))


def main():
    if len(sys.argv) < 3:
        print("Usage: python orchestrator.py <video_or_folder> <work_root>")
        sys.exit(1)
    target, work_root = sys.argv[1], sys.argv[2]
    path = Path(target)

    if path.is_dir():
        videos = sorted(path.glob("*.mkv")) + sorted(path.glob("*.mp4"))
    else:
        videos = [path]

    for v in videos:
        process_episode(str(v), work_root)


if __name__ == "__main__":
    main()
