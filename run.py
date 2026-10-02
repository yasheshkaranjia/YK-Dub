"""
Interactive runner - the easy way to use this pipeline.
Instead of remembering orchestrator.py's command-line arguments, just run:

    python run.py

...and answer two questions: where's the video (or folder of episodes),
and do you want to assign character voices before dubbing. Everything
after that runs on its own - extraction, script splitting, optional
voice setup, dubbing, and verification, one episode at a time.

Non-interactive / scripted use is also supported, with no prompts at all:

    python run.py "/path/to/Episode01.mkv" ./work
    python run.py "/path/to/season_folder" ./work

Resume-safe at two levels, same as orchestrator.py: an episode whose
.dubbed.mp4 already exists is skipped entirely, and one whose
manifest.json/translated.json already exist (extraction/script-reading
finished on an earlier run) picks up from dub_agent.py instead of
redoing a 20-30 minute Demucs pass.

This is a thin wrapper around orchestrator.py's existing logic - it
doesn't change how any agent works, it just replaces typing long paths
on the command line with a couple of prompts (or skips them entirely
in scripted mode).
"""
import importlib.util
import json
import os
import subprocess
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
WATCHDOG_MIN_EPISODES = 3  # batches of this many episodes or more get the watchdog automatically


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


def parse_selection(raw: str, count: int):
    """Turns '1,3,4-6' (or 'all') into a sorted list of 0-based indices.
    Returns None if the text isn't a valid selection for `count` items."""
    raw = raw.strip().lower().replace(" ", "")
    if raw in ("a", "all"):
        return list(range(count))
    picked = set()
    for part in raw.split(","):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            return None
        lo, hi = int(lo), int(hi) if sep else int(lo)
        if lo < 1 or hi > count or lo > hi:
            return None
        picked.update(range(lo - 1, hi))
    return sorted(picked) or None


def choose_episodes(episodes: list, work_root: Path):
    """Shows every episode found (numbered, with done/pending status) and
    lets the user pick which to dub: '1,3,4-6', or just Enter for every
    episode not dubbed yet. Picking an already-dubbed episode re-dubs it
    (after a confirmation), which is how you redo one after changing a
    character's voice. Returns the list of episodes to process."""
    def is_done(ep):
        return (work_root / ep.stem / f"{ep.stem}.dubbed.mp4").exists()

    print(f"\nFound {len(episodes)} episode(s):")
    for i, ep in enumerate(episodes, 1):
        print(f"  {i}. {ep.name}" + ("  [already dubbed]" if is_done(ep) else ""))

    pending = [ep for ep in episodes if not is_done(ep)]
    hint = f"Enter = all {len(pending)} not yet dubbed" if pending else "nothing left to dub - pick numbers to re-dub"
    while True:
        raw = input(f"\nWhich episodes to dub? e.g. 1,3,4-6 or 'all' ({hint}): ").strip()
        if not raw:
            return pending
        idx = parse_selection(raw, len(episodes))
        if idx is not None:
            break
        print(f"  didn't understand '{raw}' - use numbers 1-{len(episodes)} like 1,3,4-6")

    chosen = [episodes[i] for i in idx]
    redo = [ep for ep in chosen if is_done(ep)]
    if redo:
        print("\nThese are already dubbed and would be dubbed again (the existing file is replaced):")
        for ep in redo:
            print(f"  - {ep.name}")
        if not ask_yes_no("Re-dub them?"):
            chosen = [ep for ep in chosen if ep not in redo]
        else:
            for ep in redo:
                (work_root / ep.stem / f"{ep.stem}.dubbed.mp4").unlink()
    return chosen


def _watchdog_already_running() -> bool:
    try:
        import psutil
    except ImportError:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if any(str(a).endswith("watchdog.py") for a in (proc.info["cmdline"] or [])):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def start_watchdog(episodes: list, work_root: Path):
    """Starts watchdog.py in its own terminal window, watching just the
    episodes about to be dubbed. Returns the process (so run.py can stop it
    when the batch ends) or None if it wasn't started. Never raises - a
    watchdog that can't start must not stop the dub itself."""
    if importlib.util.find_spec("psutil") is None:
        print("[watchdog] not started - it needs psutil. Run:  pip install psutil")
        return None
    if _watchdog_already_running():
        print("[watchdog] one is already running - not starting another")
        return None
    script = Path(__file__).resolve().parent / "watchdog.py"
    cmd = [sys.executable, str(script), str(episodes[0].resolve().parent),
           str(work_root.resolve()), "--only", *[str(e.resolve()) for e in episodes]]
    try:
        if os.name == "nt":
            proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            where = "in a separate window"
        else:  # no portable "new terminal" - log to a file instead
            log = open(work_root / "watchdog.log", "a", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            where = f"in the background (log: {work_root / 'watchdog.log'})"
    except OSError as e:
        print(f"[watchdog] couldn't start ({e}) - continuing without it")
        return None
    print(f"[watchdog] {len(episodes)} episodes queued - started the watchdog {where}. "
          f"It stops by itself when the batch finishes.")
    return proc


def stop_watchdog(proc) -> None:
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass


def process_episode(video_path: Path, work_root: Path, offer_voice_setup: bool) -> None:
    stem = video_path.stem
    work_dir = work_root / stem
    out_video = work_dir / f"{stem}.dubbed.mp4"
    failed_marker = work_dir / f"{stem}.FAILED"
    manifest_path = work_dir / f"{stem}.manifest.json"
    translated_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))

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

    # Resume-safe at the STAGE level too, not just per-episode - if
    # extraction/script-reading already finished on an earlier run (their
    # JSON is sitting on disk) there's no reason to redo a 20-30 minute
    # Demucs pass just to reach the step that actually still needs doing.
    if manifest_path.exists():
        print(f"{stem}: found existing {manifest_path.name} - skipping extraction")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        heartbeat.touch(work_root, stem, "extract")
        manifest = extract_agent.run(str(video_path), str(work_dir))

    if translated_path.exists():
        print(f"{stem}: found existing {translated_path.name} - skipping script step")
    else:
        heartbeat.touch(work_root, stem, "script")
        script_agent.run(str(manifest_path))

    # Voice setup needs a subtitle file to find character names in - skip
    # the offer entirely for episodes that fell back to Whisper (no subs,
    # so no speaker names exist to assign voices to).
    if offer_voice_setup and manifest.get("subtitle_path"):
        if ask_yes_no(f"\nAssign/review character voices for '{stem}' before dubbing?"):
            configure_voices.run(manifest["subtitle_path"], manifest.get("vocals_path"))

    heartbeat.touch(work_root, stem, "dub")
    dub_agent.run(str(translated_path), str(out_video), heartbeat_root=work_root, episode_label=stem)
    dubbed_path = Path(str(translated_path).replace(".translated.json", ".dubbed.json"))

    heartbeat.touch(work_root, stem, "verify")
    verify_agent.run(str(dubbed_path))
    print(f"\n{stem}: done -> {out_video}")


def main():
    print("YK-Dub interactive runner\n")

    # Non-interactive / scripted use, e.g.:
    #   python run.py "/path/to/Episode01.mkv" ./work
    #   python run.py "/path/to/season_folder" ./work
    # Skips every prompt below: runs all pending episodes found, leaves
    # voice setup untouched (nothing to answer interactively to), and
    # auto-collects at the end - same as orchestrator.py's own
    # non-interactive path. Plain `python run.py` with no args keeps the
    # original prompt-driven flow.
    interactive = len(sys.argv) < 2
    if interactive:
        target = ask_path("Path to a video file, or a folder of episodes: ")
    else:
        target = Path(sys.argv[1])
        if not target.exists():
            print(f"Can't find '{target}' - check the path and try again.")
            sys.exit(1)

    episodes = find_episodes(target)

    if not episodes:
        print(f"No video files ({', '.join(VIDEO_EXTENSIONS)}) found at that path.")
        sys.exit(1)

    print(f"\nFound {len(episodes)} episode(s):")
    for e in episodes:
        print(f"  - {e.name}")

    if interactive:
        work_root_raw = input("\nWork/output folder [default: ./work]: ").strip().strip('"').strip("'")
        work_root = Path(work_root_raw) if work_root_raw else Path("./work")
    else:
        work_root = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("./work")
    work_root.mkdir(parents=True, exist_ok=True)

    # Check status BEFORE asking to process anything - a restart after a
    # watchdog kill, a closed laptop lid, or just "did last night's run
    # actually finish?" all used to mean starting the whole flow blind and
    # watching it silently skip already-done episodes one by one.
    def is_done(ep):
        return (work_root / ep.stem / f"{ep.stem}.dubbed.mp4").exists()

    if interactive and (len(episodes) > 1 or is_done(episodes[0])):
        # Numbered list + pick which ones (Enter = everything not done yet).
        pending = choose_episodes(episodes, work_root)
    else:
        done = [ep for ep in episodes if is_done(ep)]
        pending = [ep for ep in episodes if not is_done(ep)]
        if done:
            print(f"\n{len(done)}/{len(episodes)} already dubbed:")
            for e in done:
                print(f"  [done] {e.name}")

    if not pending:
        print("\nNothing to dub.")
        offer_collect(work_root, interactive=interactive)
        return

    print(f"\n{len(pending)} to dub:")
    for e in pending:
        print(f"  - {e.name}")

    if interactive:
        offer_voice_setup = ask_yes_no(
            "\nReview character voices before dubbing each episode? "
            "(recommended the first time you dub a new series)"
        )
        if not ask_yes_no(f"\nProcess {len(pending)} episode(s) now?", default_yes=True):
            print("Cancelled.")
            return
    else:
        offer_voice_setup = False
        print(f"\nProcessing {len(pending)} episode(s) non-interactively...")

    succeeded, failed = [], []
    watchdog_proc = (start_watchdog(pending, work_root)
                     if len(pending) >= WATCHDOG_MIN_EPISODES else None)
    try:
        for ep in pending:
            try:
                process_episode(ep, work_root, offer_voice_setup)
                succeeded.append(ep.stem)
            except Exception as e:
                # One episode's failure (a Piper/Demucs/ffmpeg call finally
                # giving up after its timeout, a corrupt source file, whatever)
                # used to take the ENTIRE overnight batch down with it - every
                # episode after the failed one never even got attempted. This
                # logs it and moves on to the next episode instead.
                print(f"\n[run] {ep.stem} FAILED: {e}")
                print(f"[run] continuing with the remaining episodes...")
                # Persist the failure so the next run skips it instead of
                # re-attempting (and re-failing) it - process_episode()
                # above already checks for this exact marker on entry.
                try:
                    (work_root / ep.stem / f"{ep.stem}.FAILED").write_text(
                        f"failed during dub: {e}\n", encoding="utf-8"
                    )
                except OSError:
                    pass
                failed.append(ep.stem)
    finally:
        stop_watchdog(watchdog_proc)

    # Report what ACTUALLY happened - counting every queued episode as
    # "processed" used to mask failures in the final summary.
    print(f"\nDone: {len(succeeded)} episode(s) dubbed -> {work_root.resolve()}")
    if failed:
        print(f"Failed: {len(failed)} - {', '.join(failed)} (re-run to retry them)")
    offer_collect(work_root, interactive=interactive)


def offer_collect(work_root: Path, interactive: bool = True) -> None:
    """Every episode's finished .dubbed.mp4 sits in its own subfolder,
    named alongside its intermediate/working files - fine for the
    pipeline, awkward for actually sitting down and watching a season.
    This copies just the finished files into one flat folder so you can
    binge them (or copy the whole folder to a phone/USB drive) without
    digging through work/<episode>/ one at a time.

    In non-interactive mode there's no one to answer a prompt, so this
    collects automatically into the default folder instead of asking -
    the same behavior orchestrator.py's own non-interactive path uses."""
    dest = None
    if interactive:
        if not ask_yes_no("\nCopy all finished episodes into one folder for easier watching?",
                           default_yes=True):
            return
        dest_raw = input(f"Destination folder [default: {work_root.parent / 'collected'}]: ").strip().strip('"').strip("'")
        dest = Path(dest_raw) if dest_raw else None
    try:
        collect_dubbed.run(str(work_root), str(dest) if dest else None)
    except SystemExit:
        pass  # collect_dubbed exits cleanly if nothing's finished yet - not a real error here


if __name__ == "__main__":
    main()
