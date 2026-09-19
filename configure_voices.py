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

All the slow work (loading the vocals track, analysing every character's
pitch across several CPU processes, with a progress bar) happens BEFORE
the first question is asked, so the prompts themselves are instant. The
results are cached next to vocals.wav (pitch_hints.json), so re-running
on the same episode skips the analysis entirely.

Run it once per episode (or once per series if the same characters
recur) - it remembers earlier choices and only asks about new names it
hasn't seen before. Press Enter on any prompt to leave that character on
whatever it's currently set to (or the default voice, if never set).

Usage:
    python configure_voices.py <subtitle.ass> [vocals.wav]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pysubs2
from pydub import AudioSegment
from tqdm import tqdm

from pitch_worker import clip_pitch_task
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
MAX_WORKERS = 8  # cap on pitch-analysis processes; more than this stops helping


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


def _label_from_hz(median_hz: float) -> str:
    if median_hz <= MALE_PITCH_MAX_HZ:
        return "male"
    if median_hz >= FEMALE_PITCH_MIN_HZ:
        return "female"
    return "uncertain"


def _cache_path(vocals_path: str) -> Path:
    return Path(vocals_path).with_name("pitch_hints.json")


def _cache_key(sub_path: str, vocals_path: str) -> dict:
    v, s = Path(vocals_path).stat(), Path(sub_path).stat()
    return {"vocals_size": v.st_size, "vocals_mtime": int(v.st_mtime),
            "sub_size": s.st_size, "sub_mtime": int(s.st_mtime),
            "lines": MAX_LINES_SAMPLED_PER_SPEAKER}


def _load_cached_hints(sub_path: str, vocals_path: str):
    try:
        data = json.loads(_cache_path(vocals_path).read_text(encoding="utf-8"))
        if data.get("key") == _cache_key(sub_path, vocals_path):
            return {name: tuple(v) for name, v in data["hints"].items()}
    except Exception:
        pass  # missing/corrupt/stale cache - just recompute
    return None


def _save_cached_hints(sub_path: str, vocals_path: str, hints: dict) -> None:
    try:
        _cache_path(vocals_path).write_text(
            json.dumps({"key": _cache_key(sub_path, vocals_path), "hints": hints},
                       ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass  # the cache is a nicety, never a reason to fail


def load_vocals(vocals_path: str):
    """Loads the vocals track once (mono). Returns None on failure."""
    try:
        audio = AudioSegment.from_wav(vocals_path)
        return audio.set_channels(1) if audio.channels != 1 else audio
    except Exception:
        return None


def compute_pitch_hints(audio: AudioSegment, speakers: dict, workers: int = None) -> dict:
    """Runs the pitch analysis for EVERY character up front, in parallel,
    with a progress bar - so the voice-assignment prompts afterwards are
    instant instead of stalling on each name.

    The main process slices each character's longest lines out of the
    (large) vocals track and hands only those small clips to the worker
    processes, so nothing big is copied between processes. One task per
    clip (not per character) keeps the workers evenly loaded and makes the
    progress bar move smoothly.

    Returns {name: (label, median_hz)}; characters with no usable audio
    are simply absent.
    """
    sample_rate = audio.frame_rate
    tasks = []
    for name, spans in speakers.items():
        longest_first = sorted(spans, key=lambda s: s[1] - s[0], reverse=True)
        for start, end in longest_first[:MAX_LINES_SAMPLED_PER_SPEAKER]:
            clip = audio[int(start * 1000):int(end * 1000)]
            tasks.append((name, np.array(clip.get_array_of_samples(), dtype=np.int16), sample_rate))

    per_speaker = {}

    def collect(result):
        name, hz = result
        if hz is not None:
            per_speaker.setdefault(name, []).append(hz)

    def run_serial(bar):
        per_speaker.clear()
        bar.reset(total=len(tasks))
        for t in tasks:
            collect(clip_pitch_task(t))
            bar.update(1)

    workers = workers or min(MAX_WORKERS, os.cpu_count() or 1)
    workers = max(1, min(workers, len(tasks) or 1))
    bar = tqdm(total=len(tasks), desc="Analysing voices", unit="clip")
    try:
        if workers == 1:
            run_serial(bar)
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(clip_pitch_task, t) for t in tasks]
                    try:
                        for f in as_completed(futures):
                            collect(f.result())
                            bar.update(1)
                    except KeyboardInterrupt:
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
            except KeyboardInterrupt:
                raise
            except Exception:
                # Pool couldn't start or crashed (locked-down machine, etc.)
                # - redo the whole thing in this process instead.
                run_serial(bar)
    finally:
        bar.close()

    hints = {}
    for name, estimates in per_speaker.items():
        median_hz = float(np.median(estimates))
        hints[name] = (_label_from_hz(median_hz), median_hz)
    return hints


def run(sub_path: str, vocals_path: str = None) -> None:
    voices = load_json(VOICES_FILE, {})
    if not voices:
        print(f"No {VOICES_FILE.name} found, or it's empty - list your available "
              "Piper voices there first (model/config/label per voice).")
        return

    alias_list = [k for k in voices.keys() if not k.startswith("_")]
    voice_map = load_json(MAP_FILE, {"_default": alias_list[0]})
    speakers = find_speakers(sub_path)

    # Step 1: do ALL the slow work up front (load vocals once, analyse every
    # character's pitch in parallel with a progress bar). Only after that
    # do we start asking questions, so the prompts below never stall.
    hints = {}
    if vocals_path and speakers:
        hints = _load_cached_hints(sub_path, vocals_path)
        if hints is not None:
            print("Using saved pitch hints for this episode.")
        else:
            hints = {}
            print("Loading vocals track...")
            vocals_audio = load_vocals(vocals_path)
            if vocals_audio is None:
                print(f"  (couldn't load {vocals_path} for pitch hints - continuing without them)")
            else:
                hints = compute_pitch_hints(vocals_audio, speakers)
                del vocals_audio  # big - free it before the interactive part
                _save_cached_hints(sub_path, vocals_path, hints)
        print()

    # Step 2: the interactive part - instant now.
    print("Available voices:")
    for i, alias in enumerate(alias_list, 1):
        print(f"  {i}. {alias} - {voices[alias].get('label', '')}")
    print(f"\nFound {len(speakers)} speaking characters. Press Enter to keep a "
          f"character's current/default voice, or type a number to change it.")
    if hints:
        print("(pitch hints below are a rough lean from the original audio, not a "
              "verdict - anime has plenty of exceptions, use your judgment)")
    print()

    for name in speakers:
        # Case-insensitive match against already-saved assignments - subs
        # write the same character as "LIAM" in one episode and "Liam" in
        # the next, and showing the saved choice as "(default)" just
        # because the case differs invites re-doing (or breaking) it.
        current = next((v for k, v in voice_map.items()
                        if k.upper() == name.upper() and not k.startswith("_")),
                       f"(default: {voice_map['_default']})")
        hint = ""
        if name in hints:
            label, hz = hints[name]
            if label == "uncertain":
                hint = f" - pitch inconclusive (~{hz:.0f}Hz)"
            else:
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
