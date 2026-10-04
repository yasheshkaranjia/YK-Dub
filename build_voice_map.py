"""Builds voice_map.json for a whole folder of episodes WITHOUT any manual
input - the non-interactive counterpart to configure_voices.py.

    python build_voice_map.py "<folder of episodes>" [--anime "Anime Name"]

How it works:
1. Extracts each episode's subtitle track and collects every speaker name
   and its line count (the same sign-filtering rules script_agent.py uses).
2. Looks the show up on AniList (free, no key) - first from the cache, then
   by searching the filename-derived title, or --anime if you pass one.
3. Matches each speaker to its AniList character (gender, age, role,
   description) and picks the best-fitting local Supertonic voice, spreading
   the cast across the 10 voices so the show doesn't collapse onto one.
4. Speakers AniList doesn't know get guessed from name words only
   ("Grandma" -> older female, "boy" -> young male...) - and truly
   unknown crowd names ("bearkins", "twins") rotate over neutral
   background voices. Nothing is ever left on _default by accident.

The result is written to voice_map.json (the ACTIVE map) and also saved as
voice_maps/<slug>.json so it survives a later `voicemaps.py` reset. Every
assignment can still be overridden afterwards with configure_voices.py -
this tool just gives you a full cast instead of an empty map.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import voice_suggest

GENDER_WORDS = {
    "female": ["grandma", "grandmother", "mother", "mom", "girl", "woman", "wife",
               "aunt", "lady", "maid", "priestess", "sister", "daughter", "queen",
               "princess", "miss", "widow"],
    "male": ["grandpa", "grandfather", "father", "dad", "boy", "man", "men", "king",
             "prince", "uncle", "husband", "sir", "lord", "brother", "son", "old man"],
}
# Names like "grandma X" also tell us the age group.
ELDER_WORDS = ["grandma", "grandmother", "grandpa", "grandfather", "elder", "old"]
YOUNG_WORDS = ["boy", "girl", "kid", "child", "son", "daughter"]

# The 10 local Supertonic voices, tagged by gender and character type.
VOICES = {
    "m_lead": "supertonic_m1",      # young male lead
    "m_authority": "supertonic_m2",  # captains, kings
    "m_soldier": "supertonic_m3",    # background military
    "m_civil": "supertonic_m4",      # middle-aged male, butlers, civilians
    "m_villain": "supertonic_m5",    # deep/sinister male
    "f_calm": "supertonic_f1",
    "f_firm": "supertonic_f2",
    "f_energetic": "supertonic_f3",
    "f_mature": "supertonic_f4",     # mother figure, older women
    "f_soft": "supertonic_f5",       # soft female, nurses, customers
}

VILLAIN_WORDS = ["villain", "bandit", "mercenary", "thug", "assassin"]
AUTHORITY_WORDS = ["king", "lord", "captain", "commander", "general", "officer",
                   "guard", "knight", "master", "mayor", "chief", "noble"]
SOFT_WORDS = ["servant", "maid", "nurse", "prostitute", "widow", "mother"]
ENERGETIC_WORDS = ["girl", "kid", "boy", "child"]

ALIAS_TO_SLOT = {}
for slot, alias in VOICES.items():
    ALIAS_TO_SLOT[alias] = slot


def name_gender(name: str):
    words = set(re.findall(r"[a-z]+", name.lower()))
    for g, ws in GENDER_WORDS.items():
        if words & set(ws):
            return g
    return None


def name_age(name: str):
    low = name.lower()
    if any(w in low for w in ELDER_WORDS):
        return "mature"
    if any(w in low for w in YOUNG_WORDS):
        return "young"
    return None


def collect_speakers(folder: Path) -> dict:
    """{speaker name: total line count} across all episodes in the folder."""
    import subprocess
    import pysubs2
    import script_agent

    counts = {}
    for video in sorted(folder.glob("*.mkv")) + sorted(folder.glob("*.mp4")):
        probe = folder / f".probe_{video.stem}.ass"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(video),
                 "-map", "0:s:0", "-c:s", "copy", str(probe)],
                check=True, capture_output=True,
            )
            subs = pysubs2.load(str(probe))
        except (subprocess.CalledProcessError, FileNotFoundError, Exception):
            print(f"  (no readable subtitle track in {video.name} - skipped)")
            continue
        finally:
            probe.unlink(missing_ok=True)
        for e in subs:
            if e.is_comment or not e.plaintext.strip() or script_agent.is_sign_event(e):
                continue
            name = (e.name or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def get_cast(anime_arg, folder: Path):
    """AniList cast for the show: --anime arg first, then the cache, then the
    title guessed from the first episode's filename."""
    videos = sorted(folder.glob("*.mkv")) or sorted(folder.glob("*.mp4"))
    guess = voice_suggest.guess_series_title(videos[0].stem) if videos else folder.name

    for candidate in ([anime_arg] if anime_arg else []) + [guess]:
        if not candidate:
            continue
        cast = voice_suggest.load_cached_cast(candidate)
        if cast:
            print(f'using cached AniList cast for "{cast["title"]}"')
            return cast
        cast, _err, offline = voice_suggest.fetch_cast(candidate)
        if cast:
            print(f'found on AniList: "{cast["title"]}" ({len(cast["characters"])} characters)')
            voice_suggest.save_cached_cast(candidate, cast)
            return cast
        if offline:
            break
    return None


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:40] or "show"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of episode videos for one show")
    ap.add_argument("--anime", default=None,
                    help="anime name to search AniList with (default: guessed "
                         "from the first episode's filename)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    counts = collect_speakers(folder)
    if not counts:
        raise SystemExit("no speaker names found in any episode's subtitles")
    print(f"collected {len(counts)} speaker names from {folder}")

    cast = get_cast(args.anime, folder)
    if not cast:
        raise SystemExit("could not get an AniList cast - pass --anime \"<name>\" to try another title")

    # AniList name -> traits
    lookup = {}
    for ch in cast["characters"]:
        for n in ch.get("names", []):
            lookup.setdefault(voice_suggest._norm(n), ch)

    used = {}  # alias -> count of speakers already on it
    voice_map = {"_default": VOICES["m_lead"]}

    def pick(gender, age, role, tags, name):
        """Choose the alias that fits best, penalising heavily used voices."""
        slots = []
        if gender == "male":
            if age == "mature":
                slots = ["m_civil"]
            elif role == "MAIN":
                slots = ["m_lead"]
            elif role == "BACKGROUND":
                slots = ["m_soldier", "m_civil"]
            elif any(t in tags for t in VILLAIN_WORDS) or "villain" in tags:
                slots = ["m_villain"]
            elif any(t in tags for t in AUTHORITY_WORDS) or "authority" in tags:
                slots = ["m_authority"]
            else:
                slots = ["m_civil", "m_lead"]
        elif gender == "female":
            if age == "mature":
                slots = ["f_mature"]
            elif role == "MAIN":
                slots = ["f_energetic"]
            elif role == "BACKGROUND":
                slots = ["f_soft", "f_mature"]
            elif "authority" in tags:
                slots = ["f_firm"]
            else:
                slots = ["f_energetic", "f_soft"]
        else:
            # Unknown gender - neutral background slots.
            slots = ["m_civil", "f_soft"]

        best = None
        for slot in slots:
            alias = VOICES[slot]
            score = -used.get(alias, 0)
            if best is None or score > best[0]:
                best = (score, alias)
        alias = best[1]
        used[alias] = used.get(alias, 0) + 1
        return alias

    for name in sorted(counts, key=lambda n: -counts[n]):
        ch = lookup.get(voice_suggest._norm(name))
        if ch:
            tr = voice_suggest._char_traits(ch)
            gender = tr["gender"] or name_gender(name)
            age = voice_suggest._age_bucket(ch.get("age")) or name_age(name)
            role = tr["role"]
            tags = set()
            desc_low = (ch.get("desc") or "").lower()
            for w in VILLAIN_WORDS + AUTHORITY_WORDS + SOFT_WORDS:
                if w in desc_low:
                    tags.add(w)
            alias = pick(gender, age, role, tags, name)
        else:
            gender = name_gender(name)
            age = name_age(name)
            if gender == "male":
                role = "BACKGROUND" if counts[name] < 15 else "SUPPORTING"
                tags = {w for w in VILLAIN_WORDS + AUTHORITY_WORDS if w in name.lower()}
                alias = pick(gender, age, role, tags, name)
            elif gender == "female":
                role = "BACKGROUND" if counts[name] < 15 else "SUPPORTING"
                tags = {w for w in SOFT_WORDS + ENERGETIC_WORDS if w in name.lower()}
                alias = pick(gender, age, role, tags, name)
            else:
                # Truly unknown (e.g. 'bearkins', 'twins') - background voice.
                alias = pick(None, None, "BACKGROUND", set(), name)
        voice_map[name] = alias

    out = Path("voice_map.json")
    out.write_text(json.dumps(voice_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} with {len(voice_map) - 1} characters + _default")
    print(Counter(v for k, v in voice_map.items() if not k.startswith("_")))

    # Save a per-show copy so voicemaps.py reset/use can bring it back later.
    store = Path("voice_maps")
    store.mkdir(exist_ok=True)
    slug = slugify(cast["title"])
    (store / f"{slug}.json").write_text(
        json.dumps(voice_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved as voice_maps/{slug}.json (python voicemaps.py use {slug})")

    # Show how many lines are actually covered, so a mostly-unmatched show
    # is visible immediately instead of after a full dub pass.
    covered = sum(counts[n] for n in counts if n in voice_map)
    print(f"voice coverage: {covered}/{sum(counts.values())} dialogue lines")


if __name__ == "__main__":
    main()
