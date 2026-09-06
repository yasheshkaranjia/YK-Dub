"""
Collects every episode's finished <name>.dubbed.mp4 out of its own
work-folder subfolder and copies them all into one flat folder, so you
can drag the whole thing onto a phone/USB drive in one go instead of
digging through a separate folder per episode.

Usage:
    python collect_dubbed.py <work_folder> [output_folder]

Example:
    python collect_dubbed.py ./final ./final_collected
"""
import shutil
import sys
from pathlib import Path

DEFAULT_OUTPUT_NAME = "collected"


def run(work_folder: str, output_folder: str = None) -> None:
    work = Path(work_folder)
    if not work.is_dir():
        print(f"'{work_folder}' isn't a folder - point this at your work/output root.")
        sys.exit(1)

    out = Path(output_folder) if output_folder else work.parent / DEFAULT_OUTPUT_NAME
    out.mkdir(parents=True, exist_ok=True)

    found = sorted(work.glob("*/*.dubbed.mp4"))
    if not found:
        print(f"No *.dubbed.mp4 files found under {work} - has anything finished dubbing yet?")
        sys.exit(1)

    print(f"Found {len(found)} dubbed episode(s). Copying to {out}...")
    for src in found:
        dest = out / src.name
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            print(f"  skip (already copied): {src.name}")
            continue
        shutil.copy2(src, dest)
        print(f"  copied: {src.name}")

    print(f"\nDone - {len(found)} episode(s) in {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_dubbed.py <work_folder> [output_folder]")
        sys.exit(1)
    work_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run(work_arg, output_arg)
