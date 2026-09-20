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

import voice_suggest
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


def _lookup_suggestions(sub_path: str, speakers: dict, voices: dict, alias_list: list,
                        voice_map: dict):
    """Optional extra: look the anime up online (AniList) and suggest a voice
    per character. Purely additive - returns ({}, {}) (and the normal prompts
    run exactly as before) if you say no or you're offline. Returns
    (suggestions, info): suggestions = {speaker: {"alias", "reason"}},
    info = {speaker: AniList facts (role/gender/age)} for every speaker AniList
    knows, so main vs supporting is visible even where no voice is suggested.
    May also pre-fill voice_map for characters with no saved voice yet if you
    say yes to that."""
    guess = voice_suggest.guess_series_title(Path(sub_path).stem)
    cast = voice_suggest.load_cached_cast(guess)
    if cast:
        print(f'Using the saved cast list for "{cast["title"]}" (from an earlier lookup).')
    else:
        answer = input(f'Look up "{guess}" online (AniList) to suggest voices? '
                       f'[Y/n, or type a different anime name]: ').strip()
        if answer.lower() in ("n", "no"):
            return {}, {}
        query = guess if answer.lower() in ("", "y", "yes") else answer
        for _attempt in range(3):
            print(f'Searching AniList for "{query}"...')
            cast, err, offline = voice_suggest.fetch_cast(query)
            if cast is None:
                print(f"  {err}")
                if offline:
                    print("  Skipping suggestions - continuing the normal way.")
                    return {}, {}
                query = input("  Type another name to try (Enter to skip): ").strip()
                if not query:
                    return {}, {}
                continue
            print(f'  Found: {cast["title"]} ({len(cast["characters"])} characters listed)')
            if input("  Is that the right anime? [Y/n]: ").strip().lower() in ("n", "no"):
                query = input("  Type another name to try (Enter to skip): ").strip()
                if not query:
                    return {}, {}
                cast = None
                continue
            break
        else:
            return {}, {}
        if cast is None:
            return {}, {}
        voice_suggest.save_cached_cast(guess, cast)

    print("\nCast listed on AniList:")
    for line in voice_suggest.cast_overview(cast):
        print(f"  {line}")
    info = voice_suggest.match_speakers(cast, speakers)
    suggestions = voice_suggest.suggest_voices(cast, speakers, voices)
    print(f"  -> {len(info)} of this episode's {len(speakers)} speakers are on that list.")
    if not suggestions:
        print("  No voice suggestions could be made - continuing the normal way "
              "(AniList info is still shown next to each name).\n")
        return {}, info

    print(f"\nSuggested voices ({len(suggestions)} of {len(speakers)} speakers - based on each "
          f"character's gender, age, role and description):")
    for name in speakers:
        if name in suggestions:
            sg = suggestions[name]
            print(f"  {name:<20} -> {sg['alias']}  ({sg['reason']})")
    print("  (openrouter voices are never auto-suggested; speakers not listed had no match.)")

    def has_saved_voice(n):
        return any(k.upper() == n.upper() and not k.startswith("_") for k in voice_map)

    fresh = [n for n in suggestions if not has_saved_voice(n)]
    kept = len(suggestions) - len(fresh)
    if fresh and input(f"\nApply these to the {len(fresh)} character(s) with no saved voice yet"
                       f"{f' (the other {kept} keep their saved voice)' if kept else ''}? "
                       f"You can still change any of them below. [y/N]: ").strip().lower() in ("y", "yes"):
        for n in fresh:
            voice_map[n] = suggestions[n]["alias"]
        print(f"  Applied {len(fresh)}.")
    print()
    return suggestions, info


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

    # Step 1b (optional): online lookup -> per-character voice suggestions.
    # Shown next to the normal prompts below; never replaces them.
    suggestions, info = {}, {}
    try:
        suggestions, info = _lookup_suggestions(sub_path, speakers, voices, alias_list, voice_map)
    except KeyboardInterrupt:
        raise
    except Exception as e:  # a lookup problem must never block voice setup
        print(f"  (voice suggestions unavailable: {e}) - continuing the normal way.\n")

    # Step 2: the interactive part - instant now.
    print("Available voices:")
    for i, alias in enumerate(alias_list, 1):
        print(f"  {i}. {alias} - {voices[alias].get('label', '')}")
    print(f"\nFound {len(speakers)} speaking characters. Press Enter to keep a "
          f"character's current/default voice, or type a number to change it"
          + (", or 's' to take the suggested voice." if suggestions else "."))
    if info:
        print("('AniList:' notes below say whether the character is a main or supporting "
              "character on the anime's AniList page.)")
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
        fact = info.get(name)
        if fact:
            bits = [voice_suggest.ROLE_TEXT.get(fact["role"], "character")]
            if fact["gender"]:
                bits.append(fact["gender"])
            if fact["age"]:
                bits.append(f"age {fact['age']}")
            hint += (f" - AniList: {', '.join(bits)}"
                     + (f" (as {fact['name']}, similar spelling)" if fact["fuzzy"] else ""))
        sg = suggestions.get(name)
        if sg and sg["alias"] != current:
            hint += f" - suggested: {alias_list.index(sg['alias']) + 1} ({sg['alias']}, 's' to take)"
        choice = input(f"{name} [{current}]{hint}: ").strip()
        if not choice:
            continue
        if choice.lower() == "s":
            if sg:
                voice_map[name] = sg["alias"]
            else:
                print(f"  no suggestion for {name} - leaving it unchanged")
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
