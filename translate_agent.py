"""
Agent 2: Translator / Comparator
Transcribes the isolated vocals with Whisper (Japanese, with timestamps),
then translates each line with a local LLM via Ollama - giving it a few
lines of surrounding context so it reads like a conversation instead of
phrase-by-phrase, and asking it to reword the line to fit naturally
within the clip's original duration. Cross-checks against the existing
subtitle track (if any) and flags lines worth a human look.

Falls back to Whisper's own built-in translation (no context, no
reworking) if Ollama isn't reachable, so the pipeline still runs.
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pysubs2
import requests
from faster_whisper import WhisperModel
from tqdm import tqdm

MODEL_SIZE = "small"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"
CONTEXT_LINES = 2
MATCH_THRESHOLD = 0.55


def load_subtitles(path):
    if not path:
        return []
    subs = pysubs2.load(path)
    return [
        {"start": e.start / 1000.0, "end": e.end / 1000.0, "text": e.plaintext.strip()}
        for e in subs if e.plaintext.strip()
    ]


def transcribe(audio_path: str, task: str, desc: str):
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, task=task, language="ja")

    results, last_pos = [], 0.0
    with tqdm(total=round(info.duration, 1), unit="s", desc=desc) as bar:
        for s in segments:
            results.append({"start": s.start, "end": s.end, "text": s.text.strip()})
            bar.update(round(s.end - last_pos, 1))
            last_pos = s.end
    return results


def ollama_available() -> bool:
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def translate_with_llm(target: str, context_before: list, duration: float) -> str:
    context_text = " / ".join(context_before[-CONTEXT_LINES:]) if context_before else "(scene start)"
    prompt = (
        "You are dubbing an anime line from Japanese into natural, spoken English.\n"
        f"Preceding dialogue for context: {context_text}\n"
        f"Line to translate: {target}\n"
        f"Reword it, if needed, to comfortably fit within about {duration:.1f} seconds of "
        "normal speaking pace, keeping the original meaning and tone.\n"
        "Reply with ONLY the final English line - no notes, no quotes, no explanation."
    )
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def find_overlapping_sub(seg, subs):
    best, best_overlap = None, 0.0
    for s in subs:
        overlap = min(seg["end"], s["end"]) - max(seg["start"], s["start"])
        if overlap > best_overlap:
            best, best_overlap = s, overlap
    return best


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def run(manifest_path: str) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    subs = load_subtitles(manifest.get("subtitle_path"))

    use_llm = ollama_available()
    if use_llm:
        raw_segments = transcribe(manifest["vocals_path"], "transcribe", "[translate] transcribing (JA)")
    else:
        print("[translate] Ollama not reachable on localhost:11434 - falling back to "
              "Whisper's built-in translation (no context, no timing rewrite). "
              "Install Ollama + `ollama pull qwen3:4b` and start it for better results.")
        raw_segments = transcribe(manifest["vocals_path"], "translate", "[translate] transcribing+translating")

    segments = []
    context_before = []
    for seg in tqdm(raw_segments, desc="[translate] translating", unit="line", disable=not use_llm):
        duration = max(seg["end"] - seg["start"], 0.3)

        if use_llm:
            try:
                final_text = translate_with_llm(seg["text"], context_before, duration)
            except requests.exceptions.RequestException as e:
                final_text = seg["text"]
                print(f"[translate] Ollama call failed ({e}) - kept raw text for this line")
            context_before.append(final_text)
            source_text = seg["text"]
        else:
            final_text = seg["text"]  # already English from Whisper's translate task
            source_text = None

        match = find_overlapping_sub(seg, subs) if subs else None
        score = similarity(final_text, match["text"]) if match else 0.0
        flag = "no_subtitle_match" if not match else ("ok" if score >= MATCH_THRESHOLD else "mismatch")

        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "japanese_text": source_text,
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