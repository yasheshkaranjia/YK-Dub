"""
Agent 3: Synthesizer + Muxer
Turns each translated line into English speech - with Piper TTS, Kokoro
(kokoro-onnx), or a mix of both depending on how each character is
configured in voices.json - time-stretches every clip to fit its subtitle
window so it lands where the original timestamp says it should, assembles
a full-length audio track, and muxes it onto the source video. The video
stream is copied (not re-encoded), so this step stays fast even on
modest hardware.
"""
import functools
import json
import re
import subprocess
import sys
import wave
from collections import Counter
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
    """Returns resolve_fn(speaker_name) -> a voice config dict, e.g.
    {"engine": "piper", "model": ..., "config": ...} or
    {"engine": "kokoro", "voice": "af_bella", "lang": "en-us"}.
    Reads voices.json (the voice registry) + voice_map.json (speaker ->
    voice alias, written by configure_voices.py) if present. Keys starting
    with '_' in voices.json (like '_kokoro', the shared Kokoro model-file
    config) are engine config, not voice aliases, and are skipped here."""
    if not VOICES_FILE.exists() or not MAP_FILE.exists():
        return lambda speaker: {"engine": "piper", "model": PIPER_MODEL, "config": PIPER_CONFIG}

    voices = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
    voice_map = json.loads(MAP_FILE.read_text(encoding="utf-8"))
    alias_voices = {k: v for k, v in voices.items() if not k.startswith("_")}
    default_alias = voice_map.get("_default", next(iter(alias_voices)))

    def resolve(speaker: str) -> dict:
        alias = voice_map.get(speaker, default_alias)
        v = alias_voices.get(alias, alias_voices[default_alias])
        cfg = dict(v)
        cfg.setdefault("engine", "piper")  # older voices.json entries have no "engine" key
        return cfg

    return resolve


resolve_voice = load_voice_lookup()


def kokoro_engine_config() -> dict:
    """Shared Kokoro model/voices-file paths + default language, read once
    from voices.json's '_kokoro' entry. Every Kokoro-voiced character
    shares the same two downloaded model files (unlike Piper, which has a
    separate .onnx per voice) - see the README's Kokoro setup section."""
    if not VOICES_FILE.exists():
        return {}
    voices = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
    return voices.get("_kokoro", {})


def validate_voices(segments: list) -> None:
    """Checks every voice actually needed for this episode is actually
    usable BEFORE synthesis starts - a missing file used to only show up
    as one silent-skip warning per affected line, deep into a run that
    could take 20+ minutes either way. Covers both engines: Piper's
    per-voice .onnx/.onnx.json files, and Kokoro's shared model+voices
    file (checked once, if any character uses it)."""
    speakers_used = {seg.get("speaker", "").strip() for seg in segments}
    checked_models, missing, needs_kokoro = set(), [], False

    for speaker in speakers_used:
        cfg = resolve_voice(speaker)
        if cfg.get("engine") == "kokoro":
            needs_kokoro = True
            if not cfg.get("voice"):
                missing.append(f"  speaker '{speaker}' is set to the kokoro engine but has no "
                               f"'voice' id in its voices.json entry")
            continue
        model, config = cfg.get("model", PIPER_MODEL), cfg.get("config", PIPER_CONFIG)
        if model in checked_models:
            continue
        checked_models.add(model)
        for path_str, label in [(model, "model"), (config, "config")]:
            if not Path(path_str).exists():
                missing.append(f"  Piper {label} file missing: {path_str}")

    if needs_kokoro:
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            missing.append("  the 'kokoro-onnx' package isn't installed - "
                            "run: pip install -r requirements-kokoro.txt")
        kcfg = kokoro_engine_config()
        for key, label in [("model", "Kokoro model file"), ("voices", "Kokoro voices file")]:
            path_str = kcfg.get(key)
            if not path_str or not Path(path_str).exists():
                missing.append(f"  {label} missing/not set: "
                                f"{path_str or '(no _kokoro entry in voices.json)'}")

    if missing:
        print("[dub] WARNING - some voice setup referenced in voice_map.json/"
              "voices.json isn't usable. Every line using it will be "
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


WORD_PATTERN = re.compile(r"[A-Za-z']+")
ELLIPSIS_PATTERN = re.compile(r"\.\.\.|\u2026")  # literal '...' or a real '…' char

# Piper's stock defaults when a voice's own .onnx.json has no opinion.
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_W = 0.8
DEFAULT_SENTENCE_SILENCE = 0.2


def normalize_shout_caps(text: str) -> str:
    """Fansub lines shouted in-universe are often written in ALL CAPS
    ('STOP!', 'GET DOWN!'). espeak/Piper's phonemizer can mistake a long
    all-caps word for an acronym and spell it out letter-by-letter
    instead of saying the word - Title-casing it here keeps the
    pronunciation correct. The actual 'shoutiness' is put back separately
    (higher noise_scale + a volume boost, see classify_tone below), not
    through casing, since casing doesn't affect Piper's output at all."""
    def fix(word):
        letters = [c for c in word if c.isalpha()]
        return word.capitalize() if len(letters) >= 3 and word.isupper() else word
    return WORD_PATTERN.sub(lambda m: fix(m.group(0)), text)


@functools.lru_cache(maxsize=None)
def voice_inference_defaults(config_path: str) -> dict:
    """Piper voice files embed their own tuned noise_scale/length_scale/
    noise_w in an 'inference' block of the .onnx.json - reading it means
    our per-line tone nudges below are relative to what THAT voice was
    actually tuned to sound best at, instead of one hardcoded number
    applied identically to every voice. Falls back to Piper's own stock
    defaults if the file is missing or doesn't have that section (older
    voice files don't always include it)."""
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        inf = cfg.get("inference", {})
        return {
            "noise_scale": inf.get("noise_scale", DEFAULT_NOISE_SCALE),
            "length_scale": inf.get("length_scale", 1.0),
            "noise_w": inf.get("noise_w", DEFAULT_NOISE_W),
        }
    except (OSError, json.JSONDecodeError):
        return {"noise_scale": DEFAULT_NOISE_SCALE, "length_scale": 1.0, "noise_w": DEFAULT_NOISE_W}


def classify_tone(raw_text: str, inner_thought: bool = False) -> dict:
    """A cheap, rule-based read on how a line is probably delivered, from
    cues that survive straight out of the subtitle text. Piper has no
    real concept of emotion - it just reads whatever it's handed the same
    way every time - so this is the lever available to make a shouted
    line, a trailing-off line, and a flat narration line actually sound
    different from each other instead of uniformly flat.

    Returns multipliers to apply ON TOP of that voice's own tuned
    defaults (see voice_inference_defaults), plus a volume trim in dB and
    a sentence-internal pause length in seconds. Cues can stack (a
    shouted line that also trails off); the caller clamps the result so
    stacking can't run away into unstable synthesis settings."""
    words = WORD_PATTERN.findall(raw_text)
    is_shout = any(len(w) >= 3 and w.isupper() for w in words)
    stripped = raw_text.rstrip()
    is_exclaim = stripped.endswith("!") and not is_shout
    is_question = stripped.endswith("?")
    is_hesitant = bool(ELLIPSIS_PATTERN.search(raw_text)) or stripped.endswith("-")

    noise_mult = noise_w_mult = length_mult = 1.0
    volume_db, sentence_silence, tags = 0.0, DEFAULT_SENTENCE_SILENCE, []

    if is_shout:
        noise_mult *= 1.18
        noise_w_mult *= 1.1
        length_mult *= 0.93
        volume_db += 3.0
        sentence_silence = 0.12
        tags.append("shout")
    elif is_exclaim:
        noise_mult *= 1.08
        length_mult *= 0.97
        volume_db += 1.5
        sentence_silence = 0.15
        tags.append("exclaim")

    if is_hesitant:
        noise_mult *= 0.93
        length_mult *= 1.08
        sentence_silence = max(sentence_silence, 0.35)
        tags.append("hesitant")

    if is_question:
        length_mult *= 1.02
        tags.append("question")

    if inner_thought:
        noise_mult *= 0.9
        length_mult *= 1.05
        volume_db -= 2.5
        tags.append("inner_thought")

    return {
        "noise_mult": noise_mult, "noise_w_mult": noise_w_mult, "length_mult": length_mult,
        # Kokoro's "speed" is a playback-rate multiplier (bigger = faster) -
        # the OPPOSITE convention from Piper's length_scale (bigger =
        # slower) - so it's derived as a reciprocal here rather than reused
        # directly, and clamped to a narrower range since Kokoro's speed
        # control is more sensitive to extreme values than Piper's.
        "kokoro_speed": max(0.7, min(1.4, 1.0 / length_mult)),
        "volume_db": volume_db, "sentence_silence": sentence_silence, "tags": tags,
    }


def synth_segment(text: str, out_wav: str, model: str = PIPER_MODEL, config: str = PIPER_CONFIG,
                   length_scale: float = 1.0, noise_scale: float = None,
                   noise_w: float = None, sentence_silence: float = None) -> None:
    text = normalize_shout_caps(clean_honorifics(clean_stutter_text(text)))
    cmd = ["piper", "--model", model, "--config", config,
           "--length-scale", str(length_scale), "--output_file", out_wav]
    if noise_scale is not None:
        cmd += ["--noise-scale", str(noise_scale)]
    if noise_w is not None:
        cmd += ["--noise-w", str(noise_w)]
    if sentence_silence is not None:
        cmd += ["--sentence-silence", str(sentence_silence)]
    subprocess.run(
        cmd, input=text, text=True, encoding="utf-8", check=True, capture_output=True,
    )


_kokoro_engine = None  # loaded once per run, not once per line - see get_kokoro_engine()


def get_kokoro_engine():
    """Lazily loads the Kokoro ONNX model + voice-style file ONCE for the
    whole run. Unlike Piper (a fresh subprocess per line, so there's
    nothing to cache), Kokoro's model load has real up-front cost - every
    line after the first reuses this same loaded engine instead of
    reloading a ~100-300MB model per line."""
    global _kokoro_engine
    if _kokoro_engine is None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise RuntimeError(
                "a character is set to the 'kokoro' engine but the kokoro-onnx "
                "package isn't installed - run: pip install -r requirements-kokoro.txt"
            ) from e
        kcfg = kokoro_engine_config()
        model_path, voices_path = kcfg.get("model"), kcfg.get("voices")
        if not model_path or not voices_path:
            raise RuntimeError(
                "no '_kokoro' entry in voices.json (needs 'model' and 'voices' paths "
                "pointing at the downloaded kokoro-v1.0*.onnx / voices-v1.0.bin files "
                "- see the README's Kokoro setup section)"
            )
        _kokoro_engine = Kokoro(model_path, voices_path)
    return _kokoro_engine


def write_wav_from_float_samples(samples: np.ndarray, sample_rate: int, out_wav: str) -> None:
    """Writes Kokoro's raw float32 [-1, 1] sample array out as a standard
    16-bit PCM wav using the stdlib `wave` module - same tool everything
    else in this file uses to read/write wavs - rather than pulling in a
    new `soundfile` dependency just for this one conversion."""
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def synth_segment_kokoro(text: str, out_wav: str, voice_cfg: dict, speed: float = 1.0) -> None:
    text = normalize_shout_caps(clean_honorifics(clean_stutter_text(text)))
    engine = get_kokoro_engine()
    lang = voice_cfg.get("lang") or kokoro_engine_config().get("lang", "en-us")
    samples, sample_rate = engine.create(text, voice=voice_cfg["voice"], speed=speed, lang=lang)
    write_wav_from_float_samples(samples, sample_rate, out_wav)


def synth_line(text: str, raw_path: Path, voice_cfg: dict, tone: dict, target_sec: float) -> None:
    """Synthesizes one line with whichever engine this speaker is assigned
    (Piper or Kokoro), then - if the natural result lands far outside the
    subtitle's timing window - resynthesizes once more at an adjusted pace
    to land closer to it naturally, before ffmpeg's atempo takes the small
    remaining gap. Same two-pass idea either way; only the pace knob
    differs (Piper: --length-scale, Kokoro: speed - and they run in
    opposite directions, see classify_tone's kokoro_speed comment)."""
    engine = voice_cfg.get("engine", "piper")

    if engine == "kokoro":
        pace = tone["kokoro_speed"]
        synth_segment_kokoro(text, str(raw_path), voice_cfg, speed=pace)
        with wave.open(str(raw_path), "rb") as w:
            natural_sec = w.getnframes() / w.getframerate()
        tempo_needed = natural_sec / target_sec
        if tempo_needed < 0.75 or tempo_needed > 1.4:
            pace = max(0.65, min(1.5, pace * tempo_needed))
            synth_segment_kokoro(text, str(raw_path), voice_cfg, speed=pace)
        return

    model, config = voice_cfg.get("model", PIPER_MODEL), voice_cfg.get("config", PIPER_CONFIG)
    defaults = voice_inference_defaults(config)
    # Tone multipliers apply ON TOP of this voice's own tuned defaults, not
    # a flat 1.0/0.667/0.8 - a shouted line synthesizes a bit faster and
    # with more vocal variation than that SAME voice's own neutral
    # baseline, rather than every line (and every voice) converging on one
    # identical setting regardless of delivery.
    noise_scale = max(0.3, min(1.1, defaults["noise_scale"] * tone["noise_mult"]))
    noise_w = max(0.3, min(1.1, defaults["noise_w"] * tone["noise_w_mult"]))
    base_length_scale = max(0.6, min(1.5, defaults["length_scale"] * tone["length_mult"]))

    synth_segment(text, str(raw_path), model, config, length_scale=base_length_scale,
                  noise_scale=noise_scale, noise_w=noise_w, sentence_silence=tone["sentence_silence"])
    with wave.open(str(raw_path), "rb") as w:
        natural_sec = w.getnframes() / w.getframerate()
    tempo_needed = natural_sec / target_sec
    # A tempo ratio this far from 1.0 sounds noticeably robotic if left
    # entirely to ffmpeg's atempo filter - resynthesize once with a
    # length_scale aimed at the target duration instead, so pace changes
    # happen at synthesis time (a person actually talking faster/slower)
    # rather than a pitch-preserving stretch doing all the work. natural_sec
    # was produced at base_length_scale (not necessarily 1.0, now that tone
    # adjusts it), so the correction scales from that actual starting point.
    if tempo_needed < 0.75 or tempo_needed > 1.4:
        length_scale = max(0.7, min(1.6, target_sec * base_length_scale / natural_sec))
        synth_segment(text, str(raw_path), model, config, length_scale=length_scale,
                      noise_scale=noise_scale, noise_w=noise_w, sentence_silence=tone["sentence_silence"])


def stretch_to_duration(in_wav: str, out_wav: str, target_sec: float, volume_db: float = 0.0) -> None:
    current = AudioSegment.from_wav(in_wav).duration_seconds
    if current <= 0:
        Path(in_wav).rename(out_wav)
        return
    tempo = max(0.5, min(2.0, current / target_sec))  # ffmpeg atempo's safe range
    filters = [f"atempo={tempo}"]
    if abs(volume_db) > 0.01:
        # A shouted or inner-thought line's volume trim, applied in the same
        # ffmpeg call as the tempo fit rather than a second subprocess call.
        filters.append(f"volume={volume_db}dB")
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_wav, "-filter:a", ",".join(filters), out_wav],
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
    tone_tag_counts = Counter()
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
            voice_cfg = resolve_voice(seg.get("speaker", "").strip())
            tone = classify_tone(seg["final_text"], seg.get("inner_thought", False))
            tone_tag_counts.update(tone["tags"])

            synth_line(seg["final_text"], raw, voice_cfg, tone, target_sec)
            stretch_to_duration(str(raw), str(fitted), target_sec, volume_db=tone["volume_db"])
        except (subprocess.CalledProcessError, RuntimeError) as e:
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
    if tone_tag_counts:
        summary = ", ".join(f"{tag}: {n}" for tag, n in tone_tag_counts.most_common())
        print(f"[dub] tone-adjusted delivery applied - {summary}")

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


@functools.lru_cache(maxsize=None)
def detect_hw_video_encoder() -> str:
    """Probes this ffmpeg build for a usable hardware H.264 encoder by
    actually running a tiny throwaway encode - not just checking whether
    `ffmpeg -encoders` LISTS it, since a full Windows ffmpeg build lists
    nearly every encoder it was compiled with regardless of what
    hardware/drivers are actually present on this machine. Tried in a
    rough order (Nvidia, then Intel, then AMD). Returns "" - meaning
    "fall back to libx264 on the CPU", the previous/default behavior -
    if none of them actually work here. Cached so this only runs once
    per process even though mux() may be called multiple times."""
    for enc in ("h264_nvenc", "h264_qsv", "h264_amf"):
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "color=size=64x64:rate=1:d=1", "-c:v", enc, "-frames:v", "1",
                 "-f", "null", "-"],
                capture_output=True, timeout=15,
            )
            if r.returncode == 0:
                return enc
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return ""


def hw_video_encode_args(encoder: str, quality: int = 20) -> list:
    """Per-encoder flags aimed at roughly matching libx264's -crf 20
    visual quality target - each hardware vendor's ffmpeg wrapper uses a
    different rate-control scheme, so 'equivalent to CRF' isn't a single
    flag name across all three."""
    if encoder == "h264_nvenc":
        # -rc vbr -cq N -b:v 0 is nvenc's closest analog to x264's -crf:
        # quality-driven variable bitrate with no hard bitrate cap.
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                "-rc", "vbr", "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "medium",
                "-global_quality", str(quality), "-pix_fmt", "yuv420p"]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp",
                "-qp_i", str(quality), "-qp_p", str(quality), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", str(quality), "-pix_fmt", "yuv420p"]


def sign_time_windows(signs_path: str, pad_sec: float = 0.15, merge_gap_sec: float = 0.5) -> list:
    """Reads the sign-only .ass file's OWN event timestamps and returns
    merged (start_sec, end_sec) windows covering everywhere sign text
    actually appears on screen. Currently unused by mux() - kept for a
    possible future segment-encode-and-concat approach (only re-encoding
    the sign windows and stream-copying the rest, then concatenating).
    That's a bigger restructure than a quick filter option: ffmpeg's
    'subtitles' filter has no per-window enable support (see mux()'s
    comment), so getting an actual speed win out of this requires
    splitting the video into segments rather than one filtered pass."""
    import pysubs2
    subs = pysubs2.load(signs_path)
    raw = sorted((max(0.0, e.start / 1000 - pad_sec), e.end / 1000 + pad_sec) for e in subs)
    if not raw:
        return []
    merged = [list(raw[0])]
    for start, end in raw[1:]:
        if start <= merged[-1][1] + merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


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
        #
        # NOTE: ffmpeg's 'subtitles' filter does NOT support the generic
        # 'enable' timeline option ("Timeline ('enable' option) not
        # supported with filter 'subtitles'") - an earlier version of this
        # function tried to use it to skip rendering outside sign windows,
        # which crashes outright. It always renders across the whole clip;
        # what actually speeds this step up is the hardware encoder below.
        filt = f"subtitles='{escape_for_ffmpeg_filter(signs_path)}'"
        hw_encoder = detect_hw_video_encoder()
        encode_args = hw_video_encode_args(hw_encoder)
        encoder_note = f"hardware encoder {hw_encoder}" if hw_encoder else "software libx264 (no usable hardware encoder found)"
        print(f"[dub] burning sign text onto the video with {encoder_note} - this re-encodes "
              f"the whole episode and is the slowest step here. Let it finish; "
              f"ffmpeg's own progress will print below.")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", str(audio_path), *sub_inputs,
             "-filter_complex", f"[0:v]{filt}[v]",
             "-map", "[v]", *audio_maps, *sub_map,
             *encode_args,
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
