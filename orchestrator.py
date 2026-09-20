"""
Orchestrator: runs the 4-agent pipeline end to end for one video, or for
every video in a folder - one episode at a time, which is kinder to a
low-end laptop than trying to run several in parallel. Queue a whole
season overnight and check results in the morning.

Resume-safe at two levels: an episode whose .dubbed.mp4 already exists is
skipped entirely, and one whose manifest.json/translated.json already
exist (extraction/script-reading finished on an earlier run) picks up
from dub_agent.py instead of redoing Demucs. One episode failing
(including timing out - see the per-step timeouts in
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
import collect_dubbed


def process_episode(video_path: str, work_root: str) -> None:
    stem = Path(video_path).stem
    work_dir = Path(work_root) / stem
    out_video = work_dir / f"{stem}.dubbed.mp4"
    failed_marker = work_dir / f"{stem}.FAILED"
    manifest_path = work_dir / f"{stem}.manifest.json"
    translated_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))

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

    # Resume-safe at the STAGE level too, not just per-episode - an episode
    # that already has its manifest/translated JSON on disk (extraction and
    # script-reading finished on a previous run, just never got to dub/verify)
    # shouldn't redo a 20-30 minute Demucs pass just to reach the step that
    # actually still needs doing.
    if manifest_path.exists():
        print(f"[orchestrator] {stem}: found existing {manifest_path.name} - skipping extraction")
    else:
        heartbeat.touch(work_root, stem, "extract")
        extract_agent.run(video_path, str(work_dir))

    if translated_path.exists():
        print(f"[orchestrator] {stem}: found existing {translated_path.name} - skipping script step")
    else:
        heartbeat.touch(work_root, stem, "script")
        script_agent.run(str(manifest_path))

    heartbeat.touch(work_root, stem, "dub")
    dub_agent.run(str(translated_path), str(out_video), heartbeat_root=work_root, episode_label=stem)
    dubbed_path = Path(str(translated_path).replace(".translated.json", ".dubbed.json"))

    heartbeat.touch(work_root, stem, "verify")
    verify_agent.run(str(dubbed_path))


def main():
    argv = sys.argv[1:]
    # Optional "--only <video> [<video> ...]" (must come last): dub just
    # those files instead of everything in the folder. run.py's episode
    # picker + watchdog.py use this so a relaunch after a kill resumes the
    # episodes you PICKED, not every other episode sitting in the folder.
    only = None
    if "--only" in argv:
        i = argv.index("--only")
        only, argv = argv[i + 1:], argv[:i]
    if len(argv) < 2:
        print("Usage: python orchestrator.py <video_or_folder> <work_root> [--only <video> ...]")
        sys.exit(1)
    target, work_root = argv[0], argv[1]
    path = Path(target)

    if only:
        videos = [Path(v) for v in only]
    elif path.is_dir():
        videos = sorted(path.glob("*.mkv")) + sorted(path.glob("*.mp4"))
    else:
        videos = [path]

    succeeded, failed = [], []
    for v in videos:
        try:
            process_episode(str(v), work_root)
            succeeded.append(v.stem)
        except Exception as e:
            # One episode's failure (a Piper/Demucs/ffmpeg call finally
            # giving up after its timeout, a corrupt source file, whatever)
            # used to take the ENTIRE overnight batch down with it - every
            # episode after the failed one never even got attempted. This
            # logs it and moves on to the next episode instead.
            print(f"\n[orchestrator] {v.stem} FAILED: {e}")
            print(f"[orchestrator] continuing with the remaining episodes...")
            # Persist the failure so the next run skips it instead of
            # re-attempting (and re-failing) it - process_episode() above
            # already checks for this exact marker on entry.
            try:
                (Path(work_root) / v.stem / f"{v.stem}.FAILED").write_text(
                    f"failed during dub: {e}\n", encoding="utf-8"
                )
            except OSError:
                pass
            failed.append(v.stem)

    # Report what ACTUALLY happened - counting every queued episode as
    # "processed" used to mask failures in the final summary.
    print(f"\nDone: {len(succeeded)} episode(s) dubbed -> {work_root}")
    if failed:
        print(f"Failed: {len(failed)} - {', '.join(failed)} (re-run to retry them)")

    # Non-interactive (this is what watchdog.py relaunches into, and what
    # a Colab/unattended run uses) - so this collects automatically rather
    # than prompting, unlike run.py's interactive version of the same step.
    try:
        collect_dubbed.run(work_root)
    except SystemExit:
        pass  # collect_dubbed exits cleanly if nothing's finished yet - not a real error here


if __name__ == "__main__":
    main()
