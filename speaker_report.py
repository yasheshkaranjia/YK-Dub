"""List every speaker in an episode's dialogue, with line counts, so the cost
of putting a given character on OpenRouter is visible before any API spend.

    python speaker_report.py "<work_root>/<episode stem>"

Line count matters because the OpenRouter free tier allows 50 requests/day
and each spoken line is one request.
"""
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import json

import script_agent


def main() -> None:
    work_dir = Path(sys.argv[1])
    stem = work_dir.name
    translated = work_dir / f"{stem}.translated.json"
    if not translated.exists():
        raise SystemExit(f"no translated JSON at {translated}")

    data = json.loads(translated.read_text(encoding="utf-8"))
    segments = data["segments"]

    ordered = OrderedDict()
    for seg in segments:
        name = (seg.get("speaker") or "").strip() or "(no speaker tag)"
        ordered.setdefault(name, []).append(seg)

    print(f"{translated.name}")
    print(f"total dialogue lines: {len(segments)}")
    print(f"distinct speakers:    {len(ordered)}")
    print()
    print(f"{'lines':>6}  {'share':>6}  speaker")
    print("-" * 60)
    for name, segs in sorted(ordered.items(), key=lambda kv: -len(kv[1])):
        share = 100.0 * len(segs) / len(segments)
        print(f"{len(segs):6d}  {share:5.1f}%  {name}")

    # Who speaks first/last is useful for picking a reference clip: a
    # character's final lines are often his most characteristic delivery.
    print()
    print("first 3 and last 3 segments overall (to locate a reference moment):")
    for seg in segments[:3] + [None] + segments[-3:]:
        if seg is None:
            print("   ...")
            continue
        print(f"   {seg['start']:7.2f}-{seg['end']:7.2f}s  "
              f"[{(seg.get('speaker') or '?'):>12}]  {seg['final_text'][:60]!r}")


if __name__ == "__main__":
    main()