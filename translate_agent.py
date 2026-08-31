"""
Agent 2: Translator / Comparator
Runs faster-whisper in translate mode on the extracted audio (Japanese
speech -> English text) and cross-checks each line against the existing
English subtitle track (if any). Every segment gets a final English line
plus a QA flag so mismatches between the sub and the spoken audio are
visible without you having to read anything yourself.
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pysubs2
from faster_whisper import WhisperModel
from tqdm import tqdm

# "base" or "small" are the realistic choices on a CPU-only low-end laptop.
# int8 keeps RAM/CPU load down further.
MODEL_SIZE = "small"
MATCH_THRESHOLD = 0.55  # below this similarity, flag the line for review


def load_subtitles(path):
    if not path:
        return []
    subs = pysubs2.load(path)
    return [
        {"start": e.start / 1000.0, "end": e.end / 1000.0, "text": e.plaintext.strip()}
        for e in subs if e.plaintext.strip()
    ]


def transcribe_translate(audio_path: str):
    print("[translate] loading Whisper model (first run downloads it once)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, task="translate", language="ja")

    results = []
    last_pos = 0.0
    with tqdm(total=round(info.duration, 1), unit="s", desc="[translate] transcribing") as bar:
        for s in segments:
            results.append({"start": s.start, "end": s.end, "text": s.text.strip()})
            bar.update(round(s.end - last_pos, 1))
            last_pos = s.end
    return results


def find_overlapping_sub(asr_seg, subs):
    best, best_overlap = None, 0.0
    for s in subs:
        overlap = min(asr_seg["end"], s["end"]) - max(asr_seg["start"], s["start"])
        if overlap > best_overlap:
            best, best_overlap = s, overlap
    return best


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def run(manifest_path: str) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    subs = load_subtitles(manifest.get("subtitle_path"))
    asr_segments = transcribe_translate(manifest["audio_path"])

    segments = []
    for seg in asr_segments:
        match = find_overlapping_sub(seg, subs) if subs else None
        if match:
            score = similarity(seg["text"], match["text"])
            flag = "ok" if score >= MATCH_THRESHOLD else "mismatch"
            final_text = match["text"]  # prefer the (human) subtitle line
        else:
            final_text, flag, score = seg["text"], "no_subtitle_match", 0.0

        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "asr_text": seg["text"],
            "subtitle_text": match["text"] if match else None,
            "final_text": final_text,
            "similarity": round(score, 3),
            "flag": flag,
        })

    out = {**manifest, "segments": segments}
    out_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = sum(1 for s in segments if s["flag"] == "mismatch")
    print(f"[translate] {len(segments)} segments, {flagged} flagged for review -> {out_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translate_agent.py <manifest.json>")
        sys.exit(1)
    run(sys.argv[1])