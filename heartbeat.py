"""
Tiny shared heartbeat helper - writes a small JSON file recording "what
step is running and when did it last make progress." watchdog.py polls
this file to detect a genuinely stuck pipeline during an unattended
overnight batch run; nothing in the actual pipeline reads it back.

Deliberately dead simple (just a timestamp + a couple of labels) so any
step can call touch() cheaply without needing to know anything about
watchdog.py's logic, and a heartbeat write failing (disk full, weird
permissions, whatever) should never be allowed to break the actual dub -
that's why every exception here is silently swallowed.
"""
import json
import time
from pathlib import Path


def path_for(work_root) -> Path:
    return Path(work_root) / "heartbeat.json"


def touch(work_root, episode: str, step: str, **extra) -> None:
    try:
        payload = {"timestamp": time.time(), "episode": episode, "step": step, **extra}
        path_for(work_root).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def read(work_root) -> dict:
    """Returns None if no heartbeat has been written yet, or the file is
    unreadable for any reason (mid-write, permissions, etc.) - the
    watchdog treats either case as 'nothing to judge yet', not as stuck."""
    try:
        return json.loads(path_for(work_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
