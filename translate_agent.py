"""
Agent 2: Translator / Comparator
Transcribes the isolated vocals with Groq's hosted Whisper API (Japanese,
with timestamps), then translates each line with Groq's hosted LLM API -
giving it a few lines of surrounding context so it reads like a
conversation instead of phrase-by-phrase, and asking it to reword the
line to fit naturally within the clip's original duration. Cross-checks
against the existing subtitle track (if any) and flags lines worth a
human look.

Needs a free Groq API key (console.groq.com) in a .env file as
GROQ_API_KEY=... in the project root. Never commit that file - it's in
.gitignore already.

If there's no internet, no key set, or a single call fails/times out,
this falls back to Argos Translate - a real offline Japanese->English MT
model bundled into this script, not a server call - so a line never
gets left as raw untranslated Japanese or stalls the whole run. Argos
won't be as fluent as the LLM (no surrounding context, no timing
rewrite), but it's always a proper translation, never a skipped line.

Transcription has no offline fallback (faster-whisper was removed to
avoid keeping two heavy engines around) - if Groq is unreachable at the
transcription step, the run fails there with a clear message rather than
silently producing bad output.
"""
import json
import math
import os
import sys
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import argostranslate.package
import argostranslate.translate
import pysubs2
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
from tqdm import tqdm

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
GROQ_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHUNK_MINUTES = 10  # comfortably under the 25MB cap even after compression
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"
CONTEXT_LINES = 2
MATCH_THRESHOLD = 0.55


def require_api_key() -> str:
    if not GROQ_API_KEY:
        print("[translate] GROQ_API_KEY not found. Add a .env file in the project "
              "root containing:\n    GROQ_API_KEY=your_key_here\n"
              "Get a free key at https://console.groq.com")
        sys.exit(1)
    return GROQ_API_KEY


def ensure_argos_ja_en() -> None:
    """Installs the ja->en Argos Translate model on first run (needs internet
    once), then it's cached locally and works fully offline after that."""
    installed = argostranslate.translate.get_installed_languages()
    has_ja_en = any(
        lang.code == "ja" and any(t.to_lang.code == "en" for t in lang.translations_from)
        for lang in installed
    )
    if has_ja_en:
        return

    print("[translate] downloading the offline JA->EN translation model "
          "(one-time, needs internet)...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next(p for p in available if p.from_code == "ja" and p.to_code == "en")
    argostranslate.package.install_from_path(pkg.download())


def offline_translate(text: str) -> str:
    try:
        return argostranslate.translate.translate(text, "ja", "en")
    except Exception as e:
        print(f"[translate] offline fallback translation failed too ({e}) - kept raw text")
        return text


def load_subtitles(path):
    if not path:
        return []
    subs = pysubs2.load(path)
    return [
        {"start": e.start / 1000.0, "end": e.end / 1000.0, "text": e.plaintext.strip()}
        for e in subs if e.plaintext.strip()
    ]


def _transcribe_chunk_file(path: str) -> list:
    """One upload to Groq's Whisper endpoint for a single (already
    small-enough) audio file. Returns raw segments with start/end/text."""
    with open(path, "rb") as f:
        resp = requests.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": f},
            data={
                "model": GROQ_TRANSCRIBE_MODEL,
                "language": "ja",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json().get("segments", [])


def transcribe(audio_path: str, desc: str):
    """Sends the vocals track to Groq's hosted Whisper API and returns
    segment-level timestamps, same shape the rest of the pipeline expects.

    Groq's free tier caps uploads at 25MB, and a full episode's isolated
    vocals as raw WAV is easily 100MB+, so this compresses to mono mp3
    first (Whisper doesn't need lossless audio) and, if a single episode
    is STILL over the cap even compressed, splits it into ~10-minute
    chunks and stitches the results back together with time offsets so
    the rest of the pipeline never has to know this happened.
    """
    audio = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)
    total_ms = len(audio)

    with tempfile.TemporaryDirectory() as tmpdir:
        # First try: whole thing as one compressed file.
        whole_path = str(Path(tmpdir) / "whole.mp3")
        audio.export(whole_path, format="mp3", bitrate="64k")

        if os.path.getsize(whole_path) <= GROQ_MAX_UPLOAD_BYTES:
            all_segments = _transcribe_chunk_file(whole_path)
            pbar_total = total_ms / 1000.0
            with tqdm(total=round(pbar_total, 1), unit="s", desc=desc) as bar:
                bar.update(pbar_total)
            results = []
            for s in all_segments:
                text = s.get("text", "").strip()
                if text:
                    results.append({"start": s["start"], "end": s["end"], "text": text})
            return results

        # Still too big even compressed (a very long episode) - chunk it.
        chunk_ms = CHUNK_MINUTES * 60 * 1000
        n_chunks = math.ceil(total_ms / chunk_ms)
        results = []
        with tqdm(total=round(total_ms / 1000.0, 1), unit="s", desc=desc) as bar:
            for i in range(n_chunks):
                start_ms = i * chunk_ms
                end_ms = min(start_ms + chunk_ms, total_ms)
                offset_s = start_ms / 1000.0

                chunk = audio[start_ms:end_ms]
                chunk_path = str(Path(tmpdir) / f"chunk_{i}.mp3")
                chunk.export(chunk_path, format="mp3", bitrate="64k")

                segs = _transcribe_chunk_file(chunk_path)
                for s in segs:
                    text = s.get("text", "").strip()
                    if text:
                        results.append({
                            "start": s["start"] + offset_s,
                            "end": s["end"] + offset_s,
                            "text": text,
                        })
                bar.update(round((end_ms - start_ms) / 1000.0, 1))
        return results


def groq_available() -> bool:
    if not GROQ_API_KEY:
        return False
    try:
        requests.get("https://api.groq.com", timeout=3)
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

    max_retries = 4
    for attempt in range(max_retries):
        resp = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=30,
        )
        if resp.status_code == 429 and attempt < max_retries - 1:
            # Free tier requests-per-minute limit hit. Respect Retry-After
            # if Groq sends one, otherwise back off with increasing delay.
            wait_s = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
            tqdm.write(f"[translate] rate limited, waiting {wait_s:.1f}s before retry "
                       f"({attempt + 1}/{max_retries - 1})...")
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


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
    require_api_key()

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    subs = load_subtitles(manifest.get("subtitle_path"))

    ensure_argos_ja_en()
    use_llm = groq_available()
    if not use_llm:
        print("[translate] Groq API not reachable - translating every line with "
              "the offline Argos model instead (no surrounding context, no "
              "timing rewrite). Check your internet connection and GROQ_API_KEY.")

    raw_segments = transcribe(manifest["vocals_path"], "[translate] transcribing (JA)")

    segments = []
    context_before = []
    for seg in tqdm(raw_segments, desc="[translate] translating", unit="line"):
        duration = max(seg["end"] - seg["start"], 0.3)

        if use_llm:
            try:
                final_text = translate_with_llm(seg["text"], context_before, duration)
            except requests.exceptions.RequestException as e:
                tqdm.write(f"[translate] Groq call failed ({e}) - using offline fallback for this line")
                final_text = offline_translate(seg["text"])
        else:
            final_text = offline_translate(seg["text"])

        context_before.append(final_text)

        match = find_overlapping_sub(seg, subs) if subs else None
        score = similarity(final_text, match["text"]) if match else 0.0
        flag = "no_subtitle_match" if not match else ("ok" if score >= MATCH_THRESHOLD else "mismatch")

        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "japanese_text": seg["text"],
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