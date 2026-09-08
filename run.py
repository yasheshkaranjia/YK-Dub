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
import heartbeat
import collect_dubbed

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
    out_video = work_dir / f"{stem}.dubbed.mp4"
    failed_marker = work_dir / f"{stem}.FAILED"

    # Resume-safe: an overnight batch that gets killed partway through
    # (by watchdog.py, a crash, you closing the laptop) shouldn't have to
    # redo every episode that already finished when you restart it.
    if out_video.exists():
        print(f"\n{stem}: already dubbed ({out_video.name} exists) - skipping")
        return
    if failed_marker.exists():
        print(f"\n{stem}: marked FAILED on a previous run ({failed_marker.name} exists) - "
              f"skipping. Delete that file if you want to retry it.")
        return

    print(f"\n{'=' * 60}\n{stem}\n{'=' * 60}")
    heartbeat.touch(work_root, stem, "extract")

    manifest = extract_agent.run(str(video_path), str(work_dir))
    manifest_path = work_dir / f"{stem}.manifest.json"

    heartbeat.touch(work_root, stem, "script")
    translated = script_agent.run(str(manifest_path))
    translated_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))

    # Voice setup needs a subtitle file to find character names in - skip
    # the offer entirely for episodes that fell back to Whisper (no subs,
    # so no speaker names exist to assign voices to).
    if offer_voice_setup and manifest.get("subtitle_path"):
        if ask_yes_no(f"\nAssign/review character voices for '{stem}' before dubbing?"):
            configure_voices.run(manifest["subtitle_path"])

    heartbeat.touch(work_root, stem, "dub")
    dub_agent.run(str(translated_path), str(out_video), heartbeat_root=work_root, episode_label=stem)
    dubbed_path = Path(str(translated_path).replace(".translated.json", ".dubbed.json"))

    heartbeat.touch(work_root, stem, "verify")
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

    # Check status BEFORE asking to process anything - a restart after a
    # watchdog kill, a closed laptop lid, or just "did last night's run
    # actually finish?" all used to mean starting the whole flow blind and
    # watching it silently skip already-done episodes one by one.
    done, pending = [], []
    for ep in episodes:
        work_dir = work_root / ep.stem
        if (work_dir / f"{ep.stem}.dubbed.mp4").exists():
            done.append(ep)
        else:
            pending.append(ep)

    if done:
        print(f"\n{len(done)}/{len(episodes)} already dubbed:")
        for e in done:
            print(f"  [done] {e.name}")

    if not pending:
        print("\nEverything here is already dubbed - nothing to do.")
        offer_collect(work_root)
        return

    print(f"\n{len(pending)} remaining:")
    for e in pending:
        print(f"  - {e.name}")

    offer_voice_setup = ask_yes_no(
        "\nReview character voices before dubbing each episode? "
        "(recommended the first time you dub a new series)"
    )

    if not ask_yes_no(f"\nProcess {len(pending)} episode(s) now?", default_yes=True):
        print("Cancelled.")
        return

    for ep in pending:
        try:
            process_episode(ep, work_root, offer_voice_setup)
        except Exception as e:
            # One episode's failure (a Piper/Demucs/ffmpeg call finally
            # giving up after its timeout, a corrupt source file, whatever)
            # used to take the ENTIRE overnight batch down with it - every
            # episode after the failed one never even got attempted. This
            # logs it and moves on to the next episode instead.
            print(f"\n[run] {ep.stem} FAILED: {e}")
            print(f"[run] continuing with the remaining episodes...")

    print(f"\nAll done. {len(pending)} episode(s) processed -> {work_root.resolve()}")
    offer_collect(work_root)


def offer_collect(work_root: Path) -> None:
    """Every episode's finished .dubbed.mp4 sits in its own subfolder,
    named alongside its intermediate/working files - fine for the
    pipeline, awkward for actually sitting down and watching a season.
    This copies just the finished files into one flat folder so you can
    binge them (or copy the whole folder to a phone/USB drive) without
    digging through work/<episode>/ one at a time."""
    if not ask_yes_no("\nCopy all finished episodes into one folder for easier watching?",
                       default_yes=True):
        return
    dest_raw = input(f"Destination folder [default: {work_root.parent / 'collected'}]: ").strip().strip('"').strip("'")
    dest = Path(dest_raw) if dest_raw else None
    collect_dubbed.run(str(work_root), str(dest) if dest else None)


if __name__ == "__main__":
    main()
