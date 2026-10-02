"""Pre-fetch voice suggestions during the extraction stage.

Runs the AniList lookup + speaker matching in parallel with Demucs
separation, so by the time the (slow) audio work finishes, the per-speaker
voice suggestions are already computed and saved to
<suggestion_path> for configure_voices.py to pick up instantly.

Rules (per the pipeline's design):
- AniList needs no API key, so no key prompt is ever needed. If a future
  engine does require one, get_key() is the single place to add that.
- Never asks "do you want to look up...". It only asks for the anime NAME
  when the filename guess fails to match anything on AniList.
- Fails silently into (None, None, []): a failed lookup must never slow
  down or break the dub.
"""
import json
import re
import sys
import threading
from pathlib import Path

import voice_suggest


def get_key(env_var: str) -> str:
    """Placeholder for future engines that need an API key. AniList needs
    none; when one does, read .env here and prompt only if truly missing."""
    return ""


def load_suggestions(suggestion_path) -> dict:
    """Reads a pre-fetched suggestions file; {} if missing/unreadable."""
    try:
        return json.loads(Path(suggestion_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def fetch_suggestions_blocking(video_stem: str, sub_path: str, speakers: dict,
                               voices: dict, suggestion_path=None) -> dict:
    """Runs the full lookup synchronously and saves the result.

    Only asks the user for an anime name if the automatic title guess
    returns nothing usable from AniList. Returns:
        {"cast": {...}, "suggestions": {...}, "info": {...}}
    or {} if nothing could be fetched (offline / no match / cancelled).
    """
    guess = voice_suggest.guess_series_title(video_stem)
    cast = voice_suggest.load_cached_cast(guess)

    if not cast:
        print(f'[suggest] searching AniList for "{guess}"...')
        cast, err, offline = voice_suggest.fetch_cast(guess)
        tries = 0
        while cast is None and not offline and tries < 3:
            # Only ever ask for the NAME - never "do you want to look up".
            name = input(f'[suggest] no match for "{guess}" - type the anime '
                         f'name to search (Enter to skip): ').strip()
            if not name:
                break
            cast, err, offline = voice_suggest.fetch_cast(name)
            if cast:
                guess = name
            tries += 1
        if cast is None:
            print(f'[suggest] {err or "no cast found"} - skipping suggestions')
            return {}
        voice_suggest.save_cached_cast(guess, cast)
    else:
        print(f'[suggest] using cached cast for "{cast["title"]}"')

    info = voice_suggest.match_speakers(cast, speakers)
    suggestions = voice_suggest.suggest_voices(cast, speakers, voices)
    result = {"cast": cast, "suggestions": suggestions, "info": info}

    if suggestion_path:
        try:
            Path(suggestion_path).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f'[suggest] saved -> {Path(suggestion_path).name}')
        except OSError:
            pass
    return result


def fetch_suggestions_async(video_stem: str, sub_path: str, speakers: dict,
                            voices: dict, suggestion_path=None):
    """Starts fetch_suggestions_blocking in a daemon thread (for use while
    Demucs runs in the background). Returns (thread, result_holder) where
    result_holder is a list you check later: [] = still running,
    [result] = done. Never raises."""
    holder = []

    def _run():
        try:
            holder.append(fetch_suggestions_blocking(
                video_stem, sub_path, speakers, voices, suggestion_path))
        except Exception as e:
            print(f"[suggest] lookup failed ({e}) - continuing without it")
            holder.append({})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, holder


def wait_for_suggestions(thread, holder, timeout: float = 60.0) -> dict:
    """Joins the async lookup; returns {} if it never finished in time."""
    if thread is None:
        return {}
    thread.join(timeout=timeout)
    return holder[0] if holder else {}
