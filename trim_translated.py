"""
Makes a fast test copy of a translated.json for listening tests - keeps
only the lines that start within the first N seconds, so dub_agent.py
only has to synthesize a handful of lines instead of a whole episode.

It does NOT trim the source video - mux() still re-encodes the full
episode either way (that's the ~4-5 minute step regardless), but the
dub/synthesis step goes from ~13 minutes down to well under a minute.
Everything after the cutoff will just be silent (no dialogue) in the
resulting .dubbed.mp4 - expected for a quick listening test, not
something to actually watch all the way through.

Usage:
    python trim_translated.py "9-9\\[EMBER] ... - 01\\[EMBER] ... - 01.translated.json"
    python trim_translated.py "...translated.json" 90     # first 90s instead of the 120s default
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print('Usage: python trim_translated.py "<...translated.json>" [seconds=120]')
        sys.exit(1)

    src = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) >= 3 else 120.0

    data = json.loads(src.read_text(encoding="utf-8"))
    kept = [seg for seg in data["segments"] if seg["start"] < seconds]
    speakers = sorted({seg.get("speaker", "").strip() for seg in kept})

    print(f"{len(kept)}/{len(data['segments'])} lines start within the first {seconds:.0f}s")
    print("Speakers who actually speak in this window:")
    for sp in speakers:
        print(f"  - {sp}")
    if not kept:
        print("\nNo lines found in that window - try a larger N.")
        sys.exit(1)

    data["segments"] = kept
    out_stem = src.stem.replace(".translated", "") + f".first{int(seconds)}s"
    out_path = src.with_name(out_stem + ".translated.json")
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    test_video_path = src.with_name(out_stem + ".TEST.mp4")

    print(f"\nwrote {out_path}")
    print("Assign ONE of the speakers listed above to \"kokoro_test\" in voice_map.json, then run:")
    print(f'  python dub_agent.py "{out_path}" "{test_video_path}"')


if __name__ == "__main__":
    main()
