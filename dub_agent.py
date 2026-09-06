"""
Agent 3: Synthesizer + Muxer
Turns each translated line into English speech with Piper TTS, time-
stretches every clip to fit its subtitle window so it lands where the
original timestamp says it should, assembles a full-length audio track,
and muxes it onto the source video. The video stream is copied (not
re-encoded), so this step stays fast even on modest hardware.
"""
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

VOICES_FILE = Path(__file__).parent / "voices.json"
MAP_FILE = Path(__file__).parent / "voice_map.json"

# Fallback if voices.json/voice_map.json don't exist yet - keeps the old
# single-voice behavior working with no setup required.
PIPER_MODEL = "en_US-lessac-medium.onnx"
PIPER_CONFIG = "en_US-lessac-medium.onnx.json"


def load_voice_lookup():
    """Returns (resolve_fn) where resolve_fn(speaker_name) -> (model, config).
    Reads voices.json (the voice registry) + voice_map.json (speaker ->
    voice alias, written by configure_voices.py) if present."""
    if not VOICES_FILE.exists() or not MAP_FILE.exists():
        return lambda speaker: (PIPER_MODEL, PIPER_CONFIG)

    voices = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
    voice_map = json.loads(MAP_FILE.read_text(encoding="utf-8"))
    default_alias = voice_map.get("_default", next(iter(voices)))

    def resolve(speaker: str):
        alias = voice_map.get(speaker, default_alias)
        v = voices.get(alias, voices[default_alias])
        return v["model"], v["config"]

    return resolve


resolve_voice = load_voice_lookup()


def validate_voices(segments: list) -> None:
    """Checks every voice actually needed for this episode has both its
    .onnx and .onnx.json present BEFORE synthesis starts - a missing
    file used to only show up as one silent-skip warning per affected
    line, deep into a run that could take 20+ minutes either way."""
    speakers_used = {seg.get("speaker", "").strip() for seg in segments}
    checked, missing = set(), []
    for speaker in speakers_used:
        model, config = resolve_voice(speaker)
        if model in checked:
            continue
        checked.add(model)
        for path_str, label in [(model, "model"), (config, "config")]:
            if not Path(path_str).exists():
                missing.append(f"  {label} file missing: {path_str}")

    if missing:
        print("[dub] WARNING - some voice files referenced in voice_map.json/"
              "voices.json don't exist on disk. Every line using them will be "
              "left silent:")
        for m in missing:
            print(m)
        print("[dub] Fix the path/download the file, or reassign that "
              "character to a different voice, before this is worth trusting.\n")


STUTTER_PATTERN = re.compile(r"\b([A-Za-z])-([A-Za-z]+)\b")
HONORIFIC_PATTERN = re.compile(r"\b(\w+)-(sama|san|kun|chan|senpai|sensei|dono)\b", re.IGNORECASE)


def clean_stutter_text(text: str) -> str:
    """Official subs write a character stammering as e.g. 'A-All' or
    'O-Oh' - Piper has no idea that's a speech hesitation and just tries
    to pronounce the literal text, hyphen included, which comes out
    broken/robotic. Swapping the hyphen for '...' when the first letter
    matches the start of the following word reads as a natural pause
    instead, without touching genuinely hyphenated words like
    'self-aware' (where the halves don't share a first letter)."""
    def replace(match):
        first_letter, word = match.group(1), match.group(2)
        if word[0].lower() == first_letter.lower():
            return f"{first_letter}... {word}"
        return match.group(0)
    return STUTTER_PATTERN.sub(replace, text)


def clean_honorifics(text: str) -> str:
    """Names sometimes keep a Japanese honorific attached with a hyphen
    (e.g. 'Liam-sama') - Piper reads the hyphen literally rather than as
    a natural break between name and title. A plain space reads far
    better without changing what's actually said."""
    return HONORIFIC_PATTERN.sub(r"\1 \2", text)


def synth_segment(text: str, out_wav: str, model: str = PIPER_MODEL, config: str = PIPER_CONFIG,
                   length_scale: float = 1.0) -> None:
    text = clean_honorifics(clean_stutter_text(text))
    subprocess.run(
        ["piper", "--model", model, "--config", config,
         "--length-scale", str(length_scale),
         "--output_file", out_wav],
        input=text, text=True, encoding="utf-8", check=True, capture_output=True,
    )


def stretch_to_duration(in_wav: str, out_wav: str, target_sec: float) -> None:
    current = AudioSegment.from_wav(in_wav).duration_seconds
    if current <= 0:
        Path(in_wav).rename(out_wav)
        return
    tempo = max(0.5, min(2.0, current / target_sec))  # ffmpeg atempo's safe range
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_wav, "-filter:a", f"atempo={tempo}", out_wav],
        check=True, capture_output=True,
    )


def build_vocal_track(segments: list, work_dir: Path, total_duration: float) -> Path:
    """Builds the full-length vocal track by adding each synthesized
    line's raw samples into one big numpy buffer at the right offset,
    instead of calling AudioSegment.overlay() per line - overlay() was
    copying the ENTIRE track's audio data on every single call, so a
    long episode with hundreds of lines could take a very long time on
    that step alone (much longer than the actual TTS synthesis it
    followed). This does the same mixing in one pass instead."""
    sample_rate, total_samples, buffer = None, None, None

    skipped = 0
    for i, seg in enumerate(tqdm(segments, desc="[dub] synthesizing lines", unit="line")):
        # A line where translation fell back to the raw source text (e.g. an
        # Ollama timeout) is still Japanese - Piper's English voice can't
        # speak it, so leave that window silent instead of crashing.
        if seg.get("japanese_text") and seg["final_text"] == seg["japanese_text"]:
            tqdm.write(f"[dub] segment {i} was never translated - leaving it silent")
            skipped += 1
            continue

        raw = work_dir / f"seg_{i:04d}_raw.wav"
        fitted = work_dir / f"seg_{i:04d}_fit.wav"
        target_sec = max(seg["end"] - seg["start"], 0.3)
        try:
            model, config = resolve_voice(seg.get("speaker", "").strip())
            synth_segment(seg["final_text"], str(raw), model, config)

            with wave.open(str(raw), "rb") as w:
                natural_sec = w.getnframes() / w.getframerate()
            tempo_needed = natural_sec / target_sec

            # A tempo ratio this far from 1.0 sounds noticeably robotic
            # if left entirely to ffmpeg's atempo filter. Piper's own
            # --length-scale changes speaking PACE at synthesis time
            # (like a person actually talking faster/slower), which
            # sounds far more natural - so for these outlier lines,
            # resynthesize once with a length_scale aimed at getting
            # close to the target, then let atempo handle only the
            # small remaining gap instead of the whole stretch.
            if tempo_needed < 0.75 or tempo_needed > 1.4:
                length_scale = max(0.7, min(1.6, target_sec / natural_sec))
                synth_segment(seg["final_text"], str(raw), model, config, length_scale=length_scale)

            stretch_to_duration(str(raw), str(fitted), target_sec)
        except subprocess.CalledProcessError as e:
            tqdm.write(f"[dub] segment {i} failed to synthesize ({e}) - leaving it silent")
            skipped += 1
            continue

        clip = AudioSegment.from_wav(fitted)
        if sample_rate is None:
            # First successful line sets the working rate for the whole
            # track - avoids either a bad hardcoded guess or resampling
            # every single line unnecessarily.
            sample_rate = clip.frame_rate
            total_samples = int(total_duration * sample_rate)
            buffer = np.zeros(total_samples, dtype=np.int32)  # int32 headroom for overlap sums
        elif clip.frame_rate != sample_rate:
            clip = clip.set_frame_rate(sample_rate)
        if clip.channels != 1:
            clip = clip.set_channels(1)
        samples = np.array(clip.get_array_of_samples(), dtype=np.int32)

        start_sample = int(seg["start"] * sample_rate)
        end_sample = min(start_sample + len(samples), total_samples)
        if start_sample >= total_samples:
            continue
        buffer[start_sample:end_sample] += samples[: end_sample - start_sample]

    if skipped:
        print(f"[dub] {skipped}/{len(segments)} lines left silent (untranslated or synth failure)")

    if buffer is None:
        # Every single line failed - still produce a valid silent track
        # rather than crash, at a reasonable default rate.
        sample_rate = 22050
        buffer = np.zeros(int(total_duration * sample_rate), dtype=np.int32)

    # Clip back to valid 16-bit range in case any overlapping lines summed past it.
    buffer = np.clip(buffer, -32768, 32767).astype(np.int16)
    track = AudioSegment(
        buffer.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1,
    )
    out_path = work_dir / "dubbed_vocals.wav"
    track.export(out_path, format="wav")
    return out_path


def mix_with_instrumental(vocals_path: Path, instrumental_path: str, work_dir: Path) -> Path:
    out_path = work_dir / "final_mix.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(vocals_path), "-i", instrumental_path,
         "-filter_complex", "amix=inputs=2:duration=longest:dropout_transition=0",
         str(out_path)],
        check=True, capture_output=True,
    )
    return out_path


def escape_for_ffmpeg_filter(path: str) -> str:
    """ffmpeg filter syntax treats ':' and '\\' specially - this makes a
    Windows path safe to drop into a subtitles= filter argument."""
    return path.replace("\\", "/").replace(":", "\\:")


def mux(video_path: str, audio_path: Path, out_path: str, signs_path: str = None,
        subtitle_path: str = None) -> None:
    # mp4 can't hold .ass subtitles directly - mov_text is the mp4-native
    # soft-subtitle format, and ffmpeg converts .ass -> mov_text on the fly.
    sub_inputs = ["-i", subtitle_path] if subtitle_path else []
    sub_map = ["-map", "2:s:0", "-c:s", "mov_text"] if subtitle_path else []

    # Audio track 0 = the new English dub (default/selected on open).
    # Audio track 1 = the original Japanese audio, kept as a second,
    # switchable track (same way subtitles are switchable) rather than
    # thrown away - copied as-is, no re-encode needed for that one.
    audio_maps = [
        "-map", "1:a:0", "-map", "0:a:0",
        "-c:a:0", "aac", "-c:a:1", "copy",
        "-metadata:s:a:0", "language=eng", "-disposition:a:0", "default",
        "-metadata:s:a:1", "language=jpn", "-disposition:a:1", "0",
    ]

    if signs_path:
        # Burning sign text onto frames means the video must be re-encoded -
        # a plain stream copy can't add pixels to existing frames. This is
        # genuinely the slowest step in the whole pipeline on CPU - can take
        # 10-30+ minutes depending on episode length and your hardware.
        # capture_output is deliberately OFF here (unlike the other ffmpeg
        # calls) so ffmpeg's own progress prints to the terminal instead of
        # going silent for that whole time - that silence was previously
        # indistinguishable from a hang.
        print("[dub] burning sign text onto the video - this re-encodes the "
              "whole episode and is the slowest step here. Let it finish; "
              "ffmpeg's own progress will print below.")
        filt = f"subtitles='{escape_for_ffmpeg_filter(signs_path)}'"
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", str(audio_path), *sub_inputs,
             "-filter_complex", f"[0:v]{filt}[v]",
             "-map", "[v]", *audio_maps, *sub_map,
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-shortest", out_path],
            check=True,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", str(audio_path), *sub_inputs,
             "-map", "0:v:0", *audio_maps, *sub_map,
             "-c:v", "copy", "-shortest", out_path],
            check=True, capture_output=True,
        )


def run(translated_manifest_path: str, out_video: str) -> dict:
    data = json.loads(Path(translated_manifest_path).read_text(encoding="utf-8"))
    work_dir = Path(translated_manifest_path).parent / "tts_work"
    work_dir.mkdir(exist_ok=True)

    validate_voices(data["segments"])
    track = build_vocal_track(data["segments"], work_dir, data["duration_sec"])
    final_mix = mix_with_instrumental(track, data["instrumental_path"], work_dir)
    mux(data["video_path"], final_mix, out_video, data.get("signs_path"), data.get("subtitle_path"))

    result = {**data, "dubbed_video": str(Path(out_video).resolve()),
              "dubbed_vocals_path": str(track.resolve())}
    result_path = Path(str(translated_manifest_path).replace(".translated.json", ".dubbed.json"))
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dub] wrote {out_video} -> manifest at {result_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dub_agent.py <translated_manifest.json> <out_video.mp4>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
