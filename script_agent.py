"""
Agent 2 (replacement for translate_agent.py): Script Reader
Fansub .ass/.srt files already contain the human translation - for
dialogue lines AND for on-screen sign/title text, which fansubbers
position and style separately from dialogue. So instead of re-translating
audio with Whisper+LLM, we just read the subtitle file directly:

- "Dialogue" lines (the normal spoken lines) become the segments that get
  dubbed by dub_agent.py.
- "Sign" lines (identified by style name or a \\pos()/\\an() position
  override - fansub convention for on-screen text) are kept as their own
  mini subtitle file, to be burned onto the video by dub_agent.py, in
  their original position/style - not dubbed.

If a video has NO subtitle track at all, this falls back to Whisper's
own Japanese->English translation for the whole audio (lower quality,
no context, but keeps the pipeline working end to end).
"""
import json
import re
import sys
from pathlib import Path

import pysubs2
from faster_whisper import WhisperModel
from tqdm import tqdm

MODEL_SIZE = "small"  # good accuracy/speed balance on CPU; step up to "medium" if a low-sub-quality fallback run sounds off, or down to "base" if it's painfully slow
SIGN_STYLE_HINTS = ("sign", "op", "ed", "title", "song", "insert", "note")
# \pos() places text at an exact screen coordinate; \an1-\an9 overrides the
# default bottom-center alignment. Both are the standard fansub convention
# for "this is an on-screen overlay, not a spoken line" - dialogue almost
# never needs to say where on screen it appears, signs always do.
POSITION_TAG = re.compile(r"\\pos\(|\\an[1-9]")


def is_sign_event(event) -> bool:
    # Two independent signals, either one is enough: an explicit style
    # name that says what it is, or a position override that implies it
    # (some fansub groups reuse the "Default" style for signs too, so
    # style name alone isn't reliable on its own).
    style = (event.style or "").lower()
    name = (event.name or "").lower()
    if any(hint in style or hint in name for hint in SIGN_STYLE_HINTS):
        return True
    return bool(POSITION_TAG.search(event.text))


def split_subtitles(sub_path: str):
    subs = pysubs2.load(sub_path)
    dialogue, signs = pysubs2.SSAFile(), pysubs2.SSAFile()
    # Both new files need the original style definitions copied over, or
    # the sign file loses its font/color/position info when re-saved -
    # pysubs2.SSAFile() starts empty, styles included.
    dialogue.styles = subs.styles
    signs.styles = subs.styles

    dialogue_segments = []
    for e in subs:
        if e.is_comment or not e.plaintext.strip():
            continue
        if is_sign_event(e):
            signs.append(e)
        else:
            dialogue.append(e)
            dialogue_segments.append({
                "start": e.start / 1000.0,  # pysubs2 uses milliseconds; everything downstream expects seconds
                "end": e.end / 1000.0,
                "final_text": e.plaintext.strip(),
                # e.name is the ASS "actor" field - fansubbers often (not
                # always) fill this in per character, which is what lets
                # configure_voices.py offer per-character voice picking.
                # Empty string when absent, never falls back to "_default"
                # here - dub_agent.py's resolve_voice() handles that.
                "speaker": (e.name or "").strip(),
            })
    return dialogue_segments, signs if len(signs) else None


def fallback_whisper_translate(audio_path: str):
    print("[script] no subtitle file found - falling back to Whisper's own "
          "translation (lower quality, no sign text to preserve)")
    # int8 quantization roughly halves memory/CPU cost vs float32, at a
    # small accuracy cost - worth it for a CPU-only fallback path that
    # ideally shouldn't even run often (subtitles are the preferred path).
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    # task="translate" (not "transcribe") makes Whisper itself do JA->EN
    # directly in one pass, since there's no subtitle translation to lean
    # on here.
    result, info = model.transcribe(audio_path, task="translate", language="ja")

    segments, last_pos = [], 0.0
    with tqdm(total=round(info.duration, 1), unit="s", desc="[script] transcribing+translating") as bar:
        for s in result:
            segments.append({"start": s.start, "end": s.end, "final_text": s.text.strip()})
            bar.update(round(s.end - last_pos, 1))
            last_pos = s.end
    return segments


def run(manifest_path: str) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    work_dir = Path(manifest_path).parent

    signs_path = None
    if manifest.get("subtitle_path"):
        segments, signs = split_subtitles(manifest["subtitle_path"])
        if signs is not None:
            signs_path = work_dir / "signs.ass"
            signs.save(str(signs_path))
        print(f"[script] read {len(segments)} dialogue lines"
              f"{f' + {len(signs)} sign lines' if signs is not None else ''} directly from subtitles")
    else:
        segments = fallback_whisper_translate(manifest["vocals_path"])

    out = {**manifest, "segments": segments, "signs_path": str(signs_path) if signs_path else None}
    out_path = Path(str(manifest_path).replace(".manifest.json", ".translated.json"))
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[script] wrote {out_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_agent.py <manifest.json>")
        sys.exit(1)
    run(sys.argv[1])
