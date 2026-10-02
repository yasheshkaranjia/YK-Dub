"""Manage per-show voice maps for YK-Dub.

dub_agent.py always reads ./voice_map.json (the ACTIVE map). To keep every show's
character names in their own file (and out of git), each show's map lives in
voice_maps/<show-slug>.json, and this tool copies maps in and out of the active file.

    python voicemaps.py list                 # show saved maps
    python voicemaps.py use <slug>           # make <slug> the active map
    python voicemaps.py save <slug>          # save the active map as <slug>
    python voicemaps.py reset                # reset the active map to the template
    python voicemaps.py check                # validate the active map

Run it from the project folder. Only the Python standard library is needed.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTIVE = ROOT / "voice_map.json"
STORE = ROOT / "voice_maps"
TEMPLATE = {"_default": "supertonic_m1"}


def fail(msg: str) -> None:
    print(f"error: {msg}")
    sys.exit(1)


def slug_ok(slug: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
        fail("slug must be lowercase letters, digits, '-' or '_' (e.g. my-show)")
    return slug


def load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"{path.name} not found")
    except json.JSONDecodeError as e:
        fail(f"{path.name} is not valid JSON ({e})")
    if not isinstance(data, dict):
        fail(f"{path.name} must contain a JSON object")
    return data


def write_active(data: dict) -> None:
    ACTIVE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def backup_active() -> None:
    if ACTIVE.exists() and ACTIVE.stat().st_size > 0:
        shutil.copy(ACTIVE, ROOT / "voice_map.previous.bak.json")


def cmd_list() -> None:
    files = sorted(STORE.glob("*.json")) if STORE.exists() else []
    if not files:
        print("no saved maps in voice_maps/")
    for f in files:
        n = len([k for k in load(f) if not k.startswith("_")])
        print(f"{f.stem}  ({n} characters)")


def cmd_use(slug: str) -> None:
    src = STORE / f"{slug_ok(slug)}.json"
    data = load(src)
    backup_active()
    write_active(data)
    print(f"active map is now '{slug}' ({len(data)} entries)")


def cmd_save(slug: str) -> None:
    data = load(ACTIVE)
    STORE.mkdir(exist_ok=True)
    dst = STORE / f"{slug_ok(slug)}.json"
    if dst.exists():
        shutil.copy(dst, STORE / f"{slug}.previous.bak")
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved active map as voice_maps/{dst.name}")


def cmd_reset() -> None:
    backup_active()
    write_active(TEMPLATE)
    print("active map reset to the template")


def cmd_check() -> None:
    data = load(ACTIVE)
    if "_default" not in data:
        fail('missing "_default" entry')
    voices = ROOT / "voices.json"
    if voices.exists():
        known = {k for k in load(voices) if not k.startswith("_")}
        bad = sorted({v for k, v in data.items() if v not in known})
        if bad:
            fail(f"aliases not found in voices.json: {', '.join(bad)}")
    chars = len([k for k in data if not k.startswith("_")])
    print(f"ok: valid JSON, {chars} characters, default = {data['_default']}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd, rest = args[0], args[1:]
    if cmd == "list":
        cmd_list()
    elif cmd == "use" and len(rest) == 1:
        cmd_use(rest[0])
    elif cmd == "save" and len(rest) == 1:
        cmd_save(rest[0])
    elif cmd == "reset":
        cmd_reset()
    elif cmd == "check":
        cmd_check()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
