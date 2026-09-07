"""
Orchestrator: runs the 4-agent pipeline end to end for one video, or for
every video in a folder - one episode at a time, which is kinder to a
low-end laptop than trying to run several in parallel. Queue a whole
season overnight and check results in the morning.

Resume-safe: an episode whose .dubbed.mp4 already exists is skipped, and
one episode failing (including timing out - see the per-step timeouts in
extract_agent.py/dub_agent.py) logs it and moves on to the next episode
rather than taking the whole batch down. Pair with watchdog.py for
unattended overnight runs - see that file's docstring.
"""
import sys
from pathlib import Path

import extract_agent
import script_agent
import dub_agent
import verify_agent
import heartbeat


def process_episode(video_path: str, work_root: str) -> None:
    stem = Path(video_path).stem
    work_dir = Path(work_root) / stem
    out_video = work_dir / f"{stem}.dubbed.mp4"
    failed_marker = work_dir / f"{stem}.FAILED"

    # Resume-safe: an overnight batch that gets killed partway through
    # (by watchdog.py, a crash, a closed laptop lid) shouldn't have to
    # redo every episode that already finished when you restart it.
    if out_video.exists():
        print(f"\n=== {stem}: already dubbed - skipping ===")
        return
    if failed_marker.exists():
        print(f"\n=== {stem}: marked FAILED on a previous run - skipping "
              f"(delete {failed_marker.name} to retry) ===")
        return

    print(f"\n=== {stem} ===")
    heartbeat.touch(work_root, stem, "extract")

    extract_agent.run(video_path, str(work_dir))
    manifest_path = work_dir / f"{stem}.manifest.json"

    heartbeat.touch(work_root, stem, "script")
    script_agent.run(str(manifest_path))
    translated_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))

    heartbeat.touch(work_root, stem, "dub")
    dub_agent.run(str(translated_path), str(out_video), heartbeat_root=work_root, episode_label=stem)
    dubbed_path = Path(str(translated_path).replace(".translated.json", ".dubbed.json"))

    heartbeat.touch(work_root, stem, "verify")
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
        try:
            process_episode(str(v), work_root)
        except Exception as e:
            # One episode's failure (a Piper/Demucs/ffmpeg call finally
            # giving up after its timeout, a corrupt source file, whatever)
            # used to take the ENTIRE overnight batch down with it - every
            # episode after the failed one never even got attempted. This
            # logs it and moves on to the next episode instead.
            print(f"\n[orchestrator] {v.stem} FAILED: {e}")
            print(f"[orchestrator] continuing with the remaining episodes...")


if __name__ == "__main__":
    main()
