"""
Interactive runner - the easy way to use this pipeline.
Instead of remembering orchestrator.py's command-line arguments, just run:

    python run.py

...and answer two questions: where's the video (or folder of episodes),
and do you want to assign character voices before dubbing. Everything
after that runs on its own - extraction, script splitting, optional
voice setup, dubbing, and verification, one episode at a time.

This is a thin wrapper around orchestrator.py's existing logic - it
doesn't change how any agent works, it just replaces typing long paths
on the command line with a couple of prompts.
"""
import sys
from pathlib import Path

import extract_agent
import script_agent
import dub_agent
import verify_agent
import configure_voices

VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".webm")


def ask_path(prompt: str) -> Path:
    """Keeps asking until the person gives a path that actually exists.
    Strips quotes, since copy-pasting a path from Windows Explorer's
    'Copy as path' often wraps it in double quotes."""
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            print("  (please enter a path)")
            continue
        path = Path(raw)
        if path.exists():
            return path
        print(f"  Can't find '{path}' - check the path and try again.")


def ask_yes_no(prompt: str, default_yes: bool = False) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    raw = input(prompt + suffix).strip().lower()
    if not raw:
        return default_yes
    return raw.startswith("y")


def find_episodes(target: Path) -> list:
    if target.is_file():
        return [target]
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(target.glob(f"*{ext}"))
    return sorted(videos)


def process_episode(video_path: Path, work_root: Path, offer_voice_setup: bool) -> None:
    stem = video_path.stem
    work_dir = work_root / stem
    print(f"\n{'=' * 60}\n{stem}\n{'=' * 60}")

    manifest = extract_agent.run(str(video_path), str(work_dir))
    manifest_path = work_dir / f"{stem}.manifest.json"

    translated = script_agent.run(str(manifest_path))
    translated_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))

    # Voice setup needs a subtitle file to find character names in - skip
    # the offer entirely for episodes that fell back to Whisper (no subs,
    # so no speaker names exist to assign voices to).
    if offer_voice_setup and manifest.get("subtitle_path"):
        if ask_yes_no(f"\nAssign/review character voices for '{stem}' before dubbing?"):
            configure_voices.run(manifest["subtitle_path"])

    out_video = work_dir / f"{stem}.dubbed.mp4"
    dub_agent.run(str(translated_path), str(out_video))
    dubbed_path = Path(str(translated_path).replace(".translated.json", ".dubbed.json"))

    verify_agent.run(str(dubbed_path))
    print(f"\n{stem}: done -> {out_video}")


def main():
    print("YK-Dub interactive runner\n")

    target = ask_path("Path to a video file, or a folder of episodes: ")
    episodes = find_episodes(target)

    if not episodes:
        print(f"No video files ({', '.join(VIDEO_EXTENSIONS)}) found at that path.")
        sys.exit(1)

    print(f"\nFound {len(episodes)} episode(s):")
    for e in episodes:
        print(f"  - {e.name}")

    work_root_raw = input("\nWork/output folder [default: ./work]: ").strip().strip('"').strip("'")
    work_root = Path(work_root_raw) if work_root_raw else Path("./work")
    work_root.mkdir(parents=True, exist_ok=True)

    offer_voice_setup = ask_yes_no(
        "\nReview character voices before dubbing each episode? "
        "(recommended the first time you dub a new series)"
    )

    if not ask_yes_no(f"\nProcess {len(episodes)} episode(s) now?", default_yes=True):
        print("Cancelled.")
        return

    for ep in episodes:
        process_episode(ep, work_root, offer_voice_setup)

    print(f"\nAll done. {len(episodes)} episode(s) processed -> {work_root.resolve()}")


if __name__ == "__main__":
    main()
