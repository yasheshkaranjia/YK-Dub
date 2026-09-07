"""
Watchdog for unattended overnight batch runs.

Run this in a SEPARATE terminal window alongside `python run.py` or
`python orchestrator.py` when queuing a long batch (a whole season) you
won't be watching. It polls the heartbeat.json file the pipeline writes
at each stage (see heartbeat.py); if that heartbeat goes stale for
longer than --stale-after minutes, it kills the pipeline's process tree
and - unless told not to - relaunches orchestrator.py to pick back up
where it left off (already-finished episodes are skipped automatically -
see orchestrator.py/run.py's resume-safe check).

Every long-running step in the pipeline (Demucs, each Piper line, the
final ffmpeg mux) now has its OWN internal timeout too (see
extract_agent.py/dub_agent.py) - those should turn a true hang into a
clean failure well before this watchdog would ever need to act. This is
a second, outer safety net for anything outside those - a Python-level
bug, or the process dying in a way that stops writing heartbeats at all.

IMPORTANT CAVEAT: Demucs' own subprocess call is one single blocking
step from this pipeline's point of view - the heartbeat can't update
DURING it, only before and after. A normal (non-stuck) Demucs run on a
slow CPU can take 25-30 minutes, so set --stale-after comfortably above
however long your slowest single step normally takes (60+ minutes is a
safer default than the 40 you might be tempted to use), or you'll get
false-positive kills on a run that was actually fine.

To avoid an infinite kill-and-relaunch loop on ONE broken episode (a
corrupted source file that hangs the exact same way every single
retry), an episode that gets killed twice in a row is marked as FAILED
(a <stem>.FAILED marker file next to its work folder) and skipped on
the next relaunch, same as a normal failure - the batch moves on
instead of retrying that one episode all night.

Usage:
    python watchdog.py <video_or_folder> <work_root> [--interval 10] [--stale-after 60] [--no-restart]

Requires: pip install psutil
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print("watchdog.py needs psutil: pip install psutil")
    sys.exit(1)

import heartbeat

STATE_FILE_NAME = "watchdog_state.json"


def find_pipeline_processes():
    """Finds any running python process whose command line mentions
    run.py or orchestrator.py - these (and their children: ffmpeg,
    demucs, piper subprocesses) are what need killing if the batch is
    genuinely stuck."""
    hits = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "run.py" in cmdline or "orchestrator.py" in cmdline:
            hits.append(proc)
    return hits


def kill_process_tree(proc) -> None:
    for child in proc.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass


def load_state(work_root: Path) -> dict:
    try:
        return json.loads((work_root / STATE_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"kill_counts": {}}


def save_state(work_root: Path, state: dict) -> None:
    (work_root / STATE_FILE_NAME).write_text(json.dumps(state), encoding="utf-8")


def mark_failed_if_repeat_offender(work_root: Path, episode: str, state: dict) -> bool:
    """Returns True if this episode has now been killed twice in a row -
    in which case it gets a .FAILED marker so orchestrator.py/run.py's
    resume-skip logic leaves it alone on the next relaunch, instead of
    the watchdog killing and relaunching into the exact same hang
    forever."""
    counts = state.setdefault("kill_counts", {})
    counts[episode] = counts.get(episode, 0) + 1
    if counts[episode] >= 2:
        work_dir = work_root / episode
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"{episode}.FAILED").write_text(
            "killed by watchdog.py twice in a row - marked failed, skipping on future runs\n",
            encoding="utf-8",
        )
        print(f"[watchdog] {episode} has now been killed {counts[episode]}x - "
              f"marking it FAILED so it's skipped instead of retried forever")
        return True
    return False


def relaunch(video_or_folder: str, work_root: str) -> None:
    print(f"[watchdog] relaunching: python orchestrator.py \"{video_or_folder}\" \"{work_root}\"")
    # Popen, not run() - the watchdog needs to keep polling while this
    # new process runs, not block waiting for it to finish.
    subprocess.Popen([sys.executable, "orchestrator.py", video_or_folder, work_root])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_or_folder", help="Same input you gave run.py/orchestrator.py")
    ap.add_argument("work_root", help="Same work/output folder you gave run.py/orchestrator.py")
    ap.add_argument("--interval", type=float, default=10, help="Minutes between checks (default 10)")
    ap.add_argument("--stale-after", type=float, default=60,
                     help="Minutes with no heartbeat update before treating the pipeline as stuck "
                          "(default 60 - see the Demucs caveat in this file's docstring)")
    ap.add_argument("--no-restart", action="store_true",
                     help="Kill a stuck run but don't relaunch it - just alert and stop watching")
    args = ap.parse_args()

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"[watchdog] watching {heartbeat.path_for(work_root)}")
    print(f"[watchdog] checking every {args.interval} min, treating "
          f"{args.stale_after}+ min of silence as stuck")

    while True:
        time.sleep(args.interval * 60)
        hb = heartbeat.read(work_root)
        if hb is None:
            print("[watchdog] no heartbeat yet - pipeline may not have started, or hasn't "
                  "reached its first checkpoint. Not acting.")
            continue

        age_min = (time.time() - hb["timestamp"]) / 60
        print(f"[watchdog] last heartbeat {age_min:.1f} min ago "
              f"(episode={hb.get('episode')}, step={hb.get('step')})")
        if age_min <= args.stale_after:
            continue

        episode = hb.get("episode") or "unknown"
        print(f"[watchdog] STUCK - {age_min:.1f} min with no progress on '{episode}' "
              f"(step: {hb.get('step')}). Killing the pipeline.")
        procs = find_pipeline_processes()
        if not procs:
            print("[watchdog] no running run.py/orchestrator.py process found - "
                  "it may have already crashed on its own.")
        for proc in procs:
            print(f"[watchdog] killing pid {proc.pid}")
            kill_process_tree(proc)

        if args.no_restart:
            print("[watchdog] --no-restart set - stopping here. Re-run the pipeline "
                  "yourself when ready; finished episodes will be skipped automatically.")
            break

        state = load_state(work_root)
        mark_failed_if_repeat_offender(work_root, episode, state)
        save_state(work_root, state)

        time.sleep(5)  # let the OS actually finish tearing down the killed processes
        relaunch(args.video_or_folder, args.work_root)


if __name__ == "__main__":
    main()
