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

MODEL_SIZE = "small"
SIGN_STYLE_HINTS = ("sign", "op", "ed", "title", "song", "insert", "note")
POSITION_TAG = re.compile(r"\\pos\(|\\an[1-9]")


def is_sign_event(event) -> bool:
    style = (event.style or "").lower()
    if any(hint in style for hint in SIGN_STYLE_HINTS):
        return True
    return bool(POSITION_TAG.search(event.text))


def split_subtitles(sub_path: str):
    subs = pysubs2.load(sub_path)
    dialogue, signs = pysubs2.SSAFile(), pysubs2.SSAFile()
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
                "start": e.start / 1000.0,
                "end": e.end / 1000.0,
                "final_text": e.plaintext.strip(),
            })
    return dialogue_segments, signs if len(signs) else None


def fallback_whisper_translate(audio_path: str):
    print("[script] no subtitle file found - falling back to Whisper's own "
          "translation (lower quality, no sign text to preserve)")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
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
