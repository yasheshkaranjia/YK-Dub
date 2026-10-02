"""Redub a specific set of episodes with the current dub_agent settings,
reusing each episode's existing extraction/subtitles instead of redoing a
20-minute Demucs pass.

    python redub.py <work_root> <episode stem> [<episode stem> ...]

Each episode is already extracted (its .manifest.json / .translated.json sit
on disk), so this only re-runs the stage that actually changed - synthesis,
mixing, muxing and verification. Used after a change to dub_agent.py (for
example the stutter effect, or the vocal-bus polish) to regenerate finished
episodes without paying for the whole pipeline again.

Existing .dubbed.mp4 / .dubbed.json are removed first, since a stale copy
would otherwise make the runner treat the episode as already finished.

UNATTENDED / OVERNIGHT USE
--------------------------
This is written to survive a whole batch without supervision:

  * One episode failing does NOT stop the rest - the exception is caught,
    printed with a full traceback, and the loop moves on to the next
    episode. (A crash partway through episode 7 used to mean episodes 8-12
    were never even attempted.)
  * Each episode is self-contained: it reads its own .translated.json and
    writes its own .dubbed.mp4, so a rerun after any interruption picks up
    exactly where it left off - finished episodes are simply redone, and
    unfinished ones get another attempt.
  * Per-episode wall-clock time is printed as it goes, so the log alone is
    enough to tell in the morning how far it got and how long each took.
  * The final summary counts what ACTUALLY succeeded rather than assuming
    every queued episode worked.
"""
import sys
import time
import traceback
from pathlib import Path

import dub_agent
import verify_agent


def redub(work_root: Path, stem: str) -> bool:
    work_dir = work_root / stem
    translated = work_dir / f"{stem}.translated.json"
    out_video = work_dir / f"{stem}.dubbed.mp4"
    dubbed_json = work_dir / f"{stem}.dubbed.json"

    if not translated.exists():
        print(f"[redub] {stem}: no {translated.name} - nothing to redub from, skipping")
        return False

    print(f"\n{'=' * 60}\n[redub] {stem}\n{'=' * 60}")
    # Delete the previous output FIRST. dub_agent.run() decides internally
    # whether it can skip (its own resume check looks at .dubbed.json), and
    # a stale file left on disk is exactly what would make it skip the work
    # we are asking for.
    for stale in (out_video, dubbed_json):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    started = time.time()
    dub_agent.run(str(translated), str(out_video), heartbeat_root=str(work_root),
                  episode_label=stem)
    verify_agent.run(str(dubbed_json))
    elapsed = time.time() - started
    size_mb = out_video.stat().st_size / (1024 * 1024) if out_video.exists() else 0.0

    # A sanity check the unattended run can act on: dub_agent.run() is
    # allowed to return without raising for some partial-failure cases, so
    # confirm the file this whole exercise exists to produce is actually
    # there and non-trivial before reporting success. An empty or missing
    # output counts as a failure, not a finished episode.
    if not out_video.exists() or size_mb < 1.0:
        raise RuntimeError(f"{out_video.name} missing or suspiciously small "
                           f"({size_mb:.1f} MB) after the dub stage")

    print(f"[redub] {stem}: done in {elapsed / 60:.1f} min -> {out_video.name} "
          f"({size_mb:.0f} MB)")
    return True


def main() -> None:
    work_root = Path(sys.argv[1])
    stems = sys.argv[2:]
    ok, failed = [], []
    batch_started = time.time()

    print(f"[redub] batch of {len(stems)} episode(s) starting - "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    for position, stem in enumerate(stems, 1):
        print(f"\n[redub] --- {position}/{len(stems)}: {stem} ---", flush=True)
        try:
            if redub(work_root, stem):
                ok.append(stem)
            # redub() returning False means "nothing to redub from" (no
            # translated JSON) - not an error, and not a success either, so
            # it is reported separately rather than counted as done.
        except Exception as e:
            # Deliberately broad. This runs overnight with nobody watching,
            # so ANY failure must be contained to this one episode and the
            # batch must continue - a missing voice file, a locked output
            # file, an ffmpeg timeout, a corrupt JSON, anything.
            print(f"[redub] {stem} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(stem)

    total_min = (time.time() - batch_started) / 60
    print(f"\n{'=' * 60}")
    print(f"[redub] BATCH FINISHED after {total_min:.1f} min - "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[redub] succeeded: {len(ok)}/{len(stems)} -> {', '.join(ok) if ok else 'none'}")
    if failed:
        print(f"[redub] FAILED: {len(failed)} -> {', '.join(failed)}")
        print("[redub] re-run just those: python redub.py \"<work_root>\" "
              + " ".join(f'"{f}"' for f in failed))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()