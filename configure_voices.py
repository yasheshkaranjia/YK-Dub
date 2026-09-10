"""
Interactive helper: scans a subtitle file for every speaking character
(sign/title lines are skipped automatically, same logic as script_agent.py)
and lets you assign each one to a Piper/Kokoro voice from voices.json.
Saves your choices to voice_map.json, which dub_agent.py reads at runtime.

If a vocals.wav path is available (extract_agent.py's Demucs output - the
isolated ORIGINAL-language vocal track), each prompt is annotated with a
rough male/female pitch lean, e.g. "LIAM [(default: lessac) - voice
sounds male, ~118Hz]:" - meant to save you from having to listen to every
character before picking, not to replace your judgment. It's a simple
autocorrelation pitch estimate over a handful of that character's lines,
not real voice classification - anime casts plenty of high-pitched male
characters and low-pitched female ones, so treat this as a nudge, not an
answer, and it deliberately never auto-picks a voice on its own.

Run it once per episode (or once per series if the same characters
recur) - it remembers earlier choices and only asks about new names it
hasn't seen before. Press Enter on any prompt to leave that character on
whatever it's currently set to (or the default voice, if never set).

Usage:
    python configure_voices.py <subtitle.ass> [vocals.wav]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pysubs2
from pydub import AudioSegment

from script_agent import is_sign_event

VOICES_FILE = Path(__file__).parent / "voices.json"
MAP_FILE = Path(__file__).parent / "voice_map.json"

# Adult speaking-voice fundamental frequency typically falls in these
# rough bands - the zone in between is genuinely ambiguous (a lot of
# real voices land there), so it's reported as "uncertain" rather than
# forced into a guess either way.
MALE_PITCH_MAX_HZ = 145
FEMALE_PITCH_MIN_HZ = 175
MAX_LINES_SAMPLED_PER_SPEAKER = 6  # keep this fast even with a large cast


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def find_speakers(sub_path: str):
    """Returns {name: [(start_sec, end_sec), ...]} for every speaking
    character, in first-appearance order - the timestamps are only used
    for the optional pitch estimate below; everything else in this file
    only needs the names."""
    subs = pysubs2.load(sub_path)
    speakers = {}
    for e in subs:
        if e.is_comment or not e.plaintext.strip() or is_sign_event(e):
            continue
        name = (e.name or "").strip()
        if not name:
            continue
        speakers.setdefault(name, []).append((e.start / 1000.0, e.end / 1000.0))
    return speakers


def estimate_pitch_hz(samples: np.ndarray, sample_rate: int, fmin: int = 70, fmax: int = 300):
    """Rough fundamental-frequency estimate via autocorrelation on one
    short voiced clip - a lean, not a real pitch tracker. Good enough to
    tell "clearly low male-range voice" from "clearly high female-range
    voice" most of the time; not something to trust on its own for the
    genuinely ambiguous middle. Kept numpy-only (no new heavy ML/audio
    dependency) so this stays fast enough to run on every speaker in a
    cast without meaningfully slowing down voice setup."""
    if len(samples) < sample_rate * 0.05:
        return None
    samples = samples.astype(np.float64)
    samples -= samples.mean()
    if np.abs(samples).max() < 1e-6:
        return None  # near-silent clip - nothing to estimate from
    corr = np.correlate(samples, samples, mode="full")[len(samples) - 1:]
    min_lag, max_lag = int(sample_rate / fmax), int(sample_rate / fmin)
    if max_lag >= len(corr) or min_lag >= max_lag:
        return None
    window = corr[min_lag:max_lag]
    if window.max() <= 0:
        return None
    peak_lag = min_lag + int(np.argmax(window))
    return sample_rate / peak_lag if peak_lag > 0 else None


def guess_gender_lean(vocals_path: str, spans: list):
    """Estimates a rough pitch across a handful of a character's longest
    lines and returns (label, median_hz) - label is 'male', 'female', or
    'uncertain'. Returns (None, None) if the audio couldn't be read or
    nothing usable was found in the sampled lines."""
    try:
        audio = AudioSegment.from_wav(vocals_path)
    except Exception:
        return None, None
    if audio.channels != 1:
        audio = audio.set_channels(1)
    sample_rate = audio.frame_rate

    longest_first = sorted(spans, key=lambda s: s[1] - s[0], reverse=True)
    estimates = []
    for start, end in longest_first[:MAX_LINES_SAMPLED_PER_SPEAKER]:
        clip = audio[int(start * 1000):int(end * 1000)]
        samples = np.array(clip.get_array_of_samples(), dtype=np.int16)
        hz = estimate_pitch_hz(samples, sample_rate)
        if hz is not None:
            estimates.append(hz)

    if not estimates:
        return None, None
    median_hz = float(np.median(estimates))
    if median_hz <= MALE_PITCH_MAX_HZ:
        return "male", median_hz
    if median_hz >= FEMALE_PITCH_MIN_HZ:
        return "female", median_hz
    return "uncertain", median_hz


def run(sub_path: str, vocals_path: str = None) -> None:
    voices = load_json(VOICES_FILE, {})
    if not voices:
        print(f"No {VOICES_FILE.name} found, or it's empty - list your available "
              "Piper voices there first (model/config/label per voice).")
        return

    alias_list = [k for k in voices.keys() if not k.startswith("_")]
    voice_map = load_json(MAP_FILE, {"_default": alias_list[0]})
    speakers = find_speakers(sub_path)

    print("Available voices:")
    for i, alias in enumerate(alias_list, 1):
        print(f"  {i}. {alias} - {voices[alias].get('label', '')}")
    print(f"\nFound {len(speakers)} speaking characters. Press Enter to keep a "
          f"character's current/default voice, or type a number to change it.")
    if vocals_path:
        print("(pitch hints below are a rough lean from the original audio, not a "
              "verdict - anime has plenty of exceptions, use your judgment)")
    print()

    for name, spans in speakers.items():
        current = voice_map.get(name, f"(default: {voice_map['_default']})")
        hint = ""
        if vocals_path:
            label, hz = guess_gender_lean(vocals_path, spans)
            if label == "uncertain":
                hint = f" - pitch inconclusive (~{hz:.0f}Hz)"
            elif label:
                hint = f" - voice sounds {label}, ~{hz:.0f}Hz"
        choice = input(f"{name} [{current}]{hint}: ").strip()
        if not choice:
            continue
        try:
            idx = int(choice) - 1
            if idx < 0:
                raise ValueError
            voice_map[name] = alias_list[idx]
        except (ValueError, IndexError):
            print(f"  didn't understand '{choice}' - leaving {name} unchanged")

    MAP_FILE.write_text(json.dumps(voice_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {MAP_FILE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python configure_voices.py <subtitle.ass> [vocals.wav]")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
