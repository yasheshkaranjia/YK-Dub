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
import os
import re
import subprocess
import sys
import threading
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

import heartbeat

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

# Piper and Kokoro don't target the same output loudness - a character
# voiced with Kokoro can come out noticeably quieter than one voiced with
# Piper even with identical tone/volume settings, just because the two
# models were trained/normalized differently. This is a FIXED per-engine
# offset, not per-line loudness normalization - it corrects a systematic
# gap between the two engines once, the same amount on every Kokoro line,
# and leaves each line's own natural dynamic variation untouched. Tune by
# ear: dub a short test clip (see trim_translated.py) with a Kokoro
# character next to a Piper one and adjust until they sit level.
ENGINE_GAIN_DB = {"kokoro": 5.0}


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


PIPER_TIMEOUT_SEC = 90  # generous for a single line - a hang here shouldn't be able to stall the whole batch


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
        timeout=PIPER_TIMEOUT_SEC,
    )


_kokoro_engine = None  # loaded once per run, not once per line - see get_kokoro_engine()
_kokoro_engine_lock = threading.Lock()


def get_kokoro_engine():
    """Lazily loads the Kokoro ONNX model + voice-style file ONCE for the
    whole run. Unlike Piper (a fresh subprocess per line, so there's
    nothing to cache), Kokoro's model load has real up-front cost - every
    line after the first reuses this same loaded engine instead of
    reloading a ~100-300MB model per line. Guarded by a lock because
    build_vocal_track() now synthesizes lines from a thread pool - without
    it, two threads could both see _kokoro_engine as None at once and
    both start loading the model concurrently."""
    global _kokoro_engine
    if _kokoro_engine is None:
        with _kokoro_engine_lock:
            if _kokoro_engine is None:  # re-check: another thread may have just finished loading
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


def find_overlapping_indices(segments: list) -> set:
    """Returns the indices of any segment whose time window overlaps
    another segment's - e.g. a narration line running over ongoing
    dialogue. Both get synthesized independently and summed directly into
    the mix in build_vocal_track(); with no ducking, two full-volume TTS
    voices summed together is just noise, not a proper dub mix (which
    would put one under the other). There's no way to tell narration from
    dialogue from subtitle text alone, so this can only mitigate it - see
    the volume_db reduction applied where this is used - not really fix
    it. The printed summary of overlap timestamps is there so you can
    find and manually judge these scenes by ear."""
    overlapping = set()
    order = sorted(range(len(segments)), key=lambda i: segments[i]["start"])
    for pos, i in enumerate(order):
        for j in order[pos + 1:]:
            if segments[j]["start"] >= segments[i]["end"]:
                break  # sorted by start - nothing further can overlap segment i
            overlapping.add(i)
            overlapping.add(j)
    return overlapping


NONVERBAL_MIN_SEC = 0.25  # shorter than this is probably a stray blip/breath, not a laugh/gasp
NONVERBAL_MAX_SEC = 3.5   # longer than this with no subtitle nearby is more likely a MISSED dialogue line
NONVERBAL_PAD_SEC = 0.2   # margin kept clear around each real subtitle line's edges
NONVERBAL_FADE_MS = 30    # a few ms fade in/out on the spliced clip avoids an audible click at the edges


def find_nonverbal_gaps(vocals_path: str, segments: list, total_duration: float) -> list:
    """Finds stretches of the ORIGINAL (Japanese) vocals track that have
    real vocal activity but no subtitle line covering them at all -
    laughs, gasps, sighs, and other non-verbal reactions are frequently
    left untranscribed in fansubs (nothing to translate), which is
    exactly why Piper/Kokoro never generates them: there's no text for
    either engine to read. Returns (start_sec, end_sec) spans, already
    excluding anything within NONVERBAL_PAD_SEC of an existing subtitle
    line's own edges (so a trailing consonant or breath right at a
    line's boundary doesn't get mistaken for a separate moment)."""
    from pydub.silence import detect_nonsilent

    audio = AudioSegment.from_wav(vocals_path)
    covered = sorted((max(0.0, s["start"] - NONVERBAL_PAD_SEC), s["end"] + NONVERBAL_PAD_SEC)
                      for s in segments)
    merged_covered = []
    for start, end in covered:
        if merged_covered and start <= merged_covered[-1][1]:
            merged_covered[-1] = (merged_covered[-1][0], max(merged_covered[-1][1], end))
        else:
            merged_covered.append((start, end))

    gaps, cursor = [], 0.0
    for start, end in merged_covered:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_duration:
        gaps.append((cursor, total_duration))

    spans = []
    for gap_start, gap_end in gaps:
        if gap_end - gap_start < NONVERBAL_MIN_SEC:
            continue
        clip = audio[int(gap_start * 1000):int(gap_end * 1000)]
        # A threshold relative to this GAP's own loudness, not a fixed dB
        # level - background music leakage in the isolated vocals track
        # varies a lot episode to episode; a fixed threshold would either
        # miss quiet laughs or trigger constantly on loud music bleed.
        hits = detect_nonsilent(clip, min_silence_len=120, silence_thresh=clip.dBFS - 16)
        for h_start, h_end in hits:
            dur = (h_end - h_start) / 1000.0
            if dur < NONVERBAL_MIN_SEC or dur > NONVERBAL_MAX_SEC:
                continue  # too short to matter, or long enough it's more likely a missed dialogue line
            spans.append((gap_start + h_start / 1000.0, gap_start + h_end / 1000.0))
    return spans


def splice_nonverbal_gaps(buffer: np.ndarray, sample_rate: int, vocals_path: str,
                           segments: list, total_duration: float) -> np.ndarray:
    """Carries over short bursts of untranslated vocal activity (laughs,
    gasps, sighs) from the original vocals track verbatim, instead of
    leaving them silent just because no subtitle line existed there for
    Piper/Kokoro to read. Trade-off worth knowing: the spliced audio is
    in that character's ORIGINAL voice-actor's timbre, not their English
    dub voice - a laugh sounding like a different voice for a second is
    far less jarring than no laugh at all, but it isn't a perfectly
    seamless match either."""
    try:
        spans = find_nonverbal_gaps(vocals_path, segments, total_duration)
    except Exception as e:
        print(f"[dub] nonverbal-gap detection skipped ({e})")
        return buffer
    if not spans:
        return buffer

    audio = AudioSegment.from_wav(vocals_path)
    if audio.channels != 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)

    total_samples = len(buffer)
    spliced = 0
    for start_sec, end_sec in spans:
        clip = audio[int(start_sec * 1000):int(end_sec * 1000)]
        if len(clip) < 10:
            continue
        fade = min(NONVERBAL_FADE_MS, len(clip) // 4)
        clip = clip.fade_in(fade).fade_out(fade)
        samples = np.array(clip.get_array_of_samples(), dtype=np.int32)
        start_sample = int(start_sec * sample_rate)
        end_sample = min(start_sample + len(samples), total_samples)
        if start_sample >= total_samples:
            continue
        buffer[start_sample:end_sample] += samples[: end_sample - start_sample]
        spliced += 1

    if spliced:
        timestamps = ", ".join(f"{s:.1f}s" for s, _ in spans[:15])
        more = "" if len(spans) <= 15 else f" (+{len(spans) - 15} more)"
        print(f"[dub] carried over {spliced} untranslated vocal burst(s) (laughs/gasps/sighs) "
              f"from the original audio - no subtitle line existed for these, so there was "
              f"nothing for Piper/Kokoro to read. Original-voice timestamps: {timestamps}{more}")
    return buffer


def build_vocal_track(segments: list, work_dir: Path, total_duration: float,
                       max_workers: int = None, heartbeat_root=None, episode_label: str = None,
                       vocals_path: str = None) -> Path:
    """Builds the full-length vocal track by synthesizing every line in
    parallel across a thread pool, then adding each finished clip's
    samples into one big numpy buffer at the right offset. Threading (not
    multiprocessing) works here because the expensive part of each line
    is subprocess.run() waiting on Piper's own process - Python releases
    the GIL while blocked on a subprocess, so many lines' waiting time
    genuinely overlaps instead of queuing behind each other one at a
    time. The actual buffer-mixing (reading a finished wav + a numpy add)
    stays single-threaded afterward - it's fast, and keeps the one piece
    of shared mutable state (the buffer) free of any concurrency bugs.

    max_workers defaults to a conservative min(4, cpu_count) - piper is
    itself lightly multi-threaded internally, so running many MORE than
    that in parallel tends to fight over the same cores rather than
    speed things up further on a modest laptop CPU."""
    max_workers = max_workers or min(4, os.cpu_count() or 2)
    sample_rate, total_samples, buffer = None, None, None
    overlapping_indices = find_overlapping_indices(segments)

    def synth_one(i: int, seg: dict):
        if seg.get("japanese_text") and seg["final_text"] == seg["japanese_text"]:
            return i, "untranslated", None
        raw = work_dir / f"seg_{i:04d}_raw.wav"
        fitted = work_dir / f"seg_{i:04d}_fit.wav"
        target_sec = max(seg["end"] - seg["start"], 0.3)
        try:
            voice_cfg = resolve_voice(seg.get("speaker", "").strip())
            tone = classify_tone(seg["final_text"], seg.get("inner_thought", False))
            synth_line(seg["final_text"], raw, voice_cfg, tone, target_sec)
            volume_db = tone["volume_db"] + ENGINE_GAIN_DB.get(voice_cfg.get("engine", "piper"), 0.0)
            if i in overlapping_indices:
                # Ducking a bit keeps two simultaneous voices from clipping
                # and softens the harshness some, though it stays two
                # voices talking at once either way - see
                # find_overlapping_indices' docstring.
                volume_db -= 3.0
            stretch_to_duration(str(raw), str(fitted), target_sec, volume_db=volume_db)
            return i, "ok", (fitted, tone["tags"])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as e:
            return i, str(e), None

    skipped = 0
    tone_tag_counts = Counter()
    results = [None] * len(segments)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(synth_one, i, seg): i for i, seg in enumerate(segments)}
        for future in tqdm(as_completed(futures), total=len(futures),
                            desc="[dub] synthesizing lines", unit="line"):
            i, status, payload = future.result()
            results[i] = (status, payload)
            completed += 1
            if heartbeat_root and completed % 10 == 0:
                heartbeat.touch(heartbeat_root, episode_label, "dub",
                                 lines_done=completed, lines_total=len(segments))

    for i, seg in enumerate(segments):
        status, payload = results[i]
        if status == "untranslated":
            tqdm.write(f"[dub] segment {i} was never translated - leaving it silent")
            skipped += 1
            continue
        if status != "ok":
            tqdm.write(f"[dub] segment {i} failed to synthesize ({status}) - leaving it silent")
            skipped += 1
            continue
        fitted, tags = payload
        tone_tag_counts.update(tags)

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
    if overlapping_indices:
        timestamps = ", ".join(f"{segments[i]['start']:.1f}s" for i in sorted(overlapping_indices)[:20])
        more = "" if len(overlapping_indices) <= 20 else f" (+{len(overlapping_indices) - 20} more)"
        print(f"[dub] {len(overlapping_indices)} line(s) overlap another line's timing window - "
              f"both get synthesized and mixed together (ducked ~3dB each, since the source "
              f"subtitles don't say which one should be the background voice). Worth listening "
              f"around: {timestamps}{more}")

    if buffer is None:
        # Every single line failed - still produce a valid silent track
        # rather than crash, at a reasonable default rate.
        sample_rate = 22050
        buffer = np.zeros(int(total_duration * sample_rate), dtype=np.int32)

    if vocals_path:
        buffer = splice_nonverbal_gaps(buffer, sample_rate, vocals_path, segments, total_duration)

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

    # No '-shortest' here on purpose - it USED to be here, and it was
    # silently truncating the whole episode. If the source .ass's last
    # dialogue line ends a couple minutes before the video does (e.g.
    # nothing spoken over the ending credits), the mov_text subtitle
    # stream's own reported duration ends there too - and '-shortest'
    # was cutting video AND audio down to match that early subtitle
    # ending, dropping the last chunk of every episode with that pattern.
    # The source video is the ground truth for how long output should be;
    # the vocal track was already built to that same length in
    # build_vocal_track(), so nothing needs '-shortest' to line up.

    with wave.open(str(audio_path), "rb") as w:
        duration_sec = w.getnframes() / w.getframerate()

    if signs_path:
        # Burning sign text onto frames means the video must be re-encoded -
        # a plain stream copy can't add pixels to existing frames. This is
        # genuinely the slowest step in the whole pipeline on CPU - can take
        # 10-30+ minutes depending on episode length and your hardware.
        # capture_output is deliberately OFF here (unlike the other ffmpeg
        # calls) so ffmpeg's own progress prints to the terminal instead of
        # going silent for that whole time - that silence was previously
        # indistinguishable from a hang. A generous timeout (6x the
        # episode's own length, floored at 30 min) is still in place below
        # it though - "no progress visible for a long time" and "actually
        # hung forever" need to stay distinguishable from each other, and
        # unbounded is how episode 6 sat "stuck" for 8+ hours unattended.
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
             out_path],
            check=True, timeout=max(1800, duration_sec * 6),
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", str(audio_path), *sub_inputs,
             "-map", "0:v:0", *audio_maps, *sub_map,
             "-c:v", "copy", out_path],
            check=True, capture_output=True, timeout=max(600, duration_sec * 2),
        )


def run(translated_manifest_path: str, out_video: str, heartbeat_root=None, episode_label: str = None) -> dict:
    data = json.loads(Path(translated_manifest_path).read_text(encoding="utf-8"))
    work_dir = Path(translated_manifest_path).parent / "tts_work"
    work_dir.mkdir(exist_ok=True)

    validate_voices(data["segments"])

    # Visibility: dub_agent.py's own progress output never says which
    # engine a line used, so a Kokoro test run and a Piper-only run look
    # identical in the console. This one line settles that question
    # up front instead of leaving you to guess from a finished video.
    speakers_used = sorted({seg.get("speaker", "").strip() for seg in data["segments"]})
    kokoro_speakers = [s for s in speakers_used if resolve_voice(s).get("engine") == "kokoro"]
    if kokoro_speakers:
        print(f"[dub] using Kokoro for: {', '.join(kokoro_speakers)} "
              f"({len(speakers_used) - len(kokoro_speakers)} other speaker(s) on Piper)")
    else:
        print(f"[dub] all {len(speakers_used)} speaker(s) on Piper - no one is assigned to Kokoro")

    track = build_vocal_track(data["segments"], work_dir, data["duration_sec"],
                               heartbeat_root=heartbeat_root, episode_label=episode_label,
                               vocals_path=data.get("vocals_path"))
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
