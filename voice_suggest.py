"""Look an anime up online (AniList - free, no account/API key) and suggest
which of YOUR voices (voices.json) fits each character.

This only ever SUGGESTS. configure_voices.py shows the suggestions next to
the normal manual prompts, so the old way (type a number per character)
works exactly as before - you can take a suggestion, ignore it, or override it.

How a suggestion is made: AniList lists each character's gender, age, role
(main/supporting/background) and a description. Those are matched against
your voices' own gender + label words (voices.json "label", e.g. "Firm
female - knights, detectives"): same gender only, then age, role, and
personality words from the description (villain, leader, cheerful, ...) pick
the closest label. Main characters get first pick of the best voices, and
voices already handed out are slightly penalised so the cast doesn't all
collapse onto one voice. Speakers AniList doesn't know (Guard, Villager A...)
only get a suggestion if their NAME says a gender (Father, Queen, Girl...),
and a character whose gender AniList doesn't give gets no suggestion (a
coin-flip helps nobody) - configure_voices still shows their main/supporting
role so you can judge yourself.

Standard library only - nothing new to install.
"""
import difflib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ANILIST_URL = "https://graphql.anilist.co"
CACHE_FILE = Path(__file__).resolve().parent / "anime_cache.json"
# Voices on these engines are never auto-suggested: they use an online API
# with its own budget/limits (see dub_agent.check_openrouter_budget), so
# spending it should be a deliberate manual choice.
SKIP_ENGINES = {"openrouter"}

QUERY = """
query ($search: String) {
  Media(search: $search, type: ANIME) {
    title { romaji english native }
    characters(perPage: 50, sort: [ROLE, RELEVANCE]) {
      edges {
        role
        node {
          name { full first last alternative }
          gender
          age
          description(asHtml: false)
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------- title guess
def guess_series_title(stem: str) -> str:
    """'[Group] Lv999 no Murabito - 01 (1080p).mkv' -> 'Lv999 no Murabito'."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", stem)
    if " " not in t.strip():
        t = re.sub(r"[._]+", " ", t)
    t = re.sub(r"\b(S\d+\s*E\d+|EP?\s*\d+|Episode\s*\d+)\b.*$", " ", t, flags=re.I)
    t = re.sub(r"\b(\d{3,4}p|x26[45]|hevc|web-?dl|web-?rip|blu-?ray|bd|aac|multi-?subs?)\b.*$", " ", t, flags=re.I)
    t = re.sub(r"\s*[-\u2013\u2014]\s*\d+(v\d+)?\s*$", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -_.")
    return t or stem


def series_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


# ---------------------------------------------------------------- AniList
def _post(query: str, variables: dict, timeout: int = 12) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        ANILIST_URL, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "YK-Dub voice suggestions"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_cast(title: str):
    """Returns (cast, error, offline). cast = {"query", "title", "characters"}
    or None. offline=True means the network itself failed (so asking the user
    to try another name would be pointless)."""
    try:
        data = _post(QUERY, {"search": title})
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, f'AniList has no anime matching "{title}".', False
        if e.code == 429:
            return None, "AniList is rate-limiting requests - try again in a minute.", True
        return None, f"AniList returned an error (HTTP {e.code}).", True
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, "Couldn't reach AniList (offline, or blocked by a firewall).", True
    except ValueError:
        return None, "AniList sent back something unreadable.", True

    media = (data or {}).get("data", {}).get("Media")
    if not media:
        return None, f'AniList has no anime matching "{title}".', False

    titles = media.get("title") or {}
    display = titles.get("english") or titles.get("romaji") or titles.get("native") or title
    if titles.get("romaji") and titles.get("english") and titles["romaji"] != titles["english"]:
        display = f'{titles["english"]} ({titles["romaji"]})'

    chars = []
    for edge in (media.get("characters") or {}).get("edges") or []:
        node = (edge or {}).get("node") or {}
        name = node.get("name") or {}
        names = [n for n in [name.get("full"), name.get("first"), name.get("last"),
                             *(name.get("alternative") or [])] if n]
        if not names:
            continue
        chars.append({
            "names": names,
            "gender": node.get("gender"),
            "age": node.get("age"),
            "role": (edge.get("role") or "SUPPORTING").upper(),
            "desc": (node.get("description") or "")[:700],
        })
    if not chars:
        return None, f'AniList found "{display}" but lists no characters for it.', False
    return {"query": title, "title": display, "characters": chars}, None, False


# ---------------------------------------------------------------- cache
def load_cached_cast(title: str):
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8")).get(series_key(title))
    except (OSError, ValueError):
        return None


def save_cached_cast(title: str, cast: dict) -> None:
    try:
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[series_key(title)] = cast
        CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # a cache is a nicety, never a reason to fail


# ---------------------------------------------------------------- profiles
_VOICE_TRAITS = {
    "young": r"\b(young|youth|teen|child|kid)\b",
    "lead": r"\b(lead|main)\b",
    "mature": r"\b(mature|middle-aged|elder|elderly|old|mother|father|butler)\b",
    "authority": r"\b(authority|captain|king|queen|commander|commanding|firm|leader|knight|detective)\b",
    "villain": r"\b(villain|sinister|deep|antagonist|evil)\b",
    "calm": r"\b(calm|soft|gentle|quiet|nurse|goddess|android)\b",
    "energetic": r"\b(energetic|lively|cheerful|bright)\b",
    "background": r"\b(crew|soldiers?|background|civilians?|customers?|generic)\b",
}
_CHAR_TRAITS = {
    "villain": r"\b(villain|antagonist|evil|demon lord|dark lord|tyrant|ruthless|sadistic|cruel)\b",
    "authority": r"\b(captain|king|queen|emperor|empress|general|commander|leader|guild ?master|chief|headmaster|knight|president)\b",
    "calm": r"\b(calm|quiet|gentle|composed|serene|soft-spoken|polite)\b",
    "energetic": r"\b(cheerful|energetic|lively|hyper|genki|bubbly|spirited|enthusiastic|playful|mischievous)\b",
}
# Speaker names that give a gender away even though AniList doesn't list them.
_GENDER_WORDS = {
    "male": {"father", "dad", "son", "boy", "man", "men", "king", "prince", "brother",
             "sir", "lord", "grandfather", "uncle", "husband", "gentleman"},
    "female": {"mother", "mom", "daughter", "girl", "woman", "women", "queen", "princess",
               "sister", "lady", "maid", "priestess", "grandmother", "aunt", "wife"},
}
# ...and the age they imply (used only for those name-only guesses).
_AGE_WORDS = {
    "adult": {"father", "dad", "mother", "mom", "uncle", "aunt", "husband", "wife", "king",
              "queen", "lord", "lady", "sir", "gentleman"},
    "mature": {"grandfather", "grandmother", "elder"},
    "young": {"son", "boy", "girl", "daughter", "prince", "princess"},
}


def voice_profile(alias: str, cfg: dict):
    """(gender or None, set of trait words) for one voices.json entry.
    An explicit "gender": "male"/"female" in the entry wins; otherwise it's
    read from the alias (..._m1 / ..._f2) or the label text."""
    label = (cfg.get("label") or "").lower()
    gender = str(cfg.get("gender") or "").lower() or None
    if gender not in ("male", "female"):
        gender = None
        m = re.search(r"(?:^|_)([mf])\d+$", alias.lower())
        if m:
            gender = "male" if m.group(1) == "m" else "female"
        elif re.search(r"\b(female|woman|girl|maid|mother)\b", label):
            gender = "female"
        elif re.search(r"\b(male|man|boy|father)\b", label):
            gender = "male"
    traits = {name for name, rx in _VOICE_TRAITS.items() if re.search(rx, label)}
    return gender, traits


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _age_bucket(age):
    m = re.search(r"\d+", str(age or ""))
    if not m:
        return None
    n = int(m.group())
    return "child" if n < 13 else "young" if n < 25 else "adult" if n < 45 else "mature"


def _char_traits(ch: dict) -> dict:
    desc = re.sub(r"~!.*?!~", " ", ch.get("desc") or "", flags=re.S).lower()  # drop spoiler blocks
    gender = {"male": "male", "female": "female"}.get(str(ch.get("gender") or "").lower())
    if gender is None:  # fall back to the pronouns the description uses
        he = len(re.findall(r"\b(he|his|him)\b", desc))
        she = len(re.findall(r"\b(she|her|hers)\b", desc))
        if he - she >= 3:
            gender = "male"
        elif she - he >= 3:
            gender = "female"
    tags = {name for name, rx in _CHAR_TRAITS.items() if re.search(rx, desc)}
    return {"gender": gender, "age": _age_bucket(ch.get("age")), "role": ch.get("role", "SUPPORTING"),
            "tags": tags}


def _score(c: dict, v_gender, v_traits: set, used: int):
    """Higher = better fit; None = never (wrong gender)."""
    s = 0.0
    if c["gender"] and v_gender:
        if c["gender"] != v_gender:
            return None
        s += 10
    elif c["gender"] and not v_gender:
        s -= 2
    age, role, tags = c["age"], c["role"], c["tags"]
    if age == "child" and "young" in v_traits:
        s += 2
    elif age == "young":
        s += 3 if ({"young", "lead"} & v_traits) else 0
        s -= 2 if "mature" in v_traits else 0
    elif age == "adult":
        s += 1 if "mature" in v_traits else 0
        s -= 1 if "young" in v_traits else 0
    elif age == "mature":
        s += 3 if "mature" in v_traits else 0
        s -= 2 if ({"young", "lead"} & v_traits) else 0
    if role == "MAIN":
        s += 3 if "lead" in v_traits else 0
        s -= 3 if "background" in v_traits else 0
    elif role == "BACKGROUND":
        s += 2 if "background" in v_traits else 0
        s -= 2 if "lead" in v_traits else 0
    else:
        s -= 2 if "lead" in v_traits else 0  # keep the lead voice for the lead
    for t, w in (("villain", 4), ("authority", 3), ("calm", 2), ("energetic", 3)):
        if t in tags and t in v_traits:
            s += w
    if "energetic" in tags and "calm" in v_traits:
        s -= 1
    return s - 1.5 * used


def _reason(c: dict, matched: str, via_name: bool) -> str:
    bits = [c["gender"] or "gender unknown"]
    if c["age"]:
        bits.append({"child": "child", "young": "young", "adult": "adult", "mature": "older"}[c["age"]])
    if not via_name:
        bits.append({"MAIN": "main character", "SUPPORTING": "supporting",
                     "BACKGROUND": "minor"}.get(c["role"], "supporting"))
        bits.extend(sorted(c["tags"]))
        return f'{matched}: ' + ", ".join(bits)
    return "name suggests " + (c["gender"] or "?") + (f", {c['age']}" if c["age"] else "")


ROLE_TEXT = {"MAIN": "main character", "SUPPORTING": "supporting character",
             "BACKGROUND": "minor character"}


def _build_lookup(cast: dict) -> dict:
    """normalised name -> character (AniList lists main characters first, so
    the first one wins when two characters share a name)."""
    lookup = {}
    for ch in cast.get("characters", []):
        for n in ch.get("names", []):
            lookup.setdefault(_norm(n), ch)
    return lookup


def _find_character(lookup: dict, speaker: str):
    """(character, is_fuzzy). Exact name first; then a close spelling match
    (subtitle romanisation often differs from AniList's by a letter, e.g.
    Lizel/Rizel) - flagged so the user knows it's a guess."""
    key = _norm(speaker)
    if key in lookup:
        return lookup[key], False
    if len(key) >= 5:
        close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.8)
        if close:
            return lookup[close[0]], True
    return None, False


def match_speakers(cast: dict, speakers: dict) -> dict:
    """{speaker: {"name", "role", "gender", "age", "fuzzy"}} for every subtitle
    speaker that is a character AniList knows - so the user can see who is a
    main/supporting character even where no voice can be suggested."""
    lookup = _build_lookup(cast)
    out = {}
    for speaker in speakers:
        ch, fuzzy = _find_character(lookup, speaker)
        if ch:
            out[speaker] = {"name": ch["names"][0], "role": ch.get("role", "SUPPORTING"),
                            "gender": (ch.get("gender") or "").lower() or None,
                            "age": ch.get("age") or None, "fuzzy": fuzzy}
    return out


def cast_overview(cast: dict, main_limit: int = 8, support_limit: int = 10) -> list:
    """Printable lines: the main and supporting characters AniList lists."""
    def fmt(ch):
        bits = [b for b in [(ch.get("gender") or "").lower(), str(ch["age"]) if ch.get("age") else ""] if b]
        return ch["names"][0] + (f" ({', '.join(bits)})" if bits else "")
    lines = []
    for role, label, limit in (("MAIN", "Main", main_limit), ("SUPPORTING", "Supporting", support_limit)):
        chars = [c for c in cast.get("characters", []) if c.get("role") == role]
        if chars:
            more = f" (+{len(chars) - limit} more)" if len(chars) > limit else ""
            lines.append(f"{label}: " + ", ".join(fmt(c) for c in chars[:limit]) + more)
    return lines


def suggest_voices(cast: dict, speakers: dict, voices: dict) -> dict:
    """{speaker_name: {"alias", "reason"}} for the speakers we can say
    something about. `speakers` is configure_voices.find_speakers() output
    ({name: [(start, end), ...]}); `voices` is voices.json."""
    profiles = {a: voice_profile(a, cfg) for a, cfg in voices.items()
                if not a.startswith("_") and isinstance(cfg, dict)
                and cfg.get("engine", "piper") not in SKIP_ENGINES}
    if not profiles:
        return {}

    lookup = _build_lookup(cast)

    todo = []  # (rank, -line_count, name, traits, matched_label, via_name)
    for name, spans in speakers.items():
        ch, fuzzy = _find_character(lookup, name)
        if ch:
            tr = _char_traits(ch)
            if tr["gender"] is None:
                continue  # gender unknown -> any voice would be a coin flip; the
                          # role/age note is still shown, just no suggestion
            todo.append(({"MAIN": 0, "SUPPORTING": 1}.get(tr["role"], 2), -len(spans),
                         name, tr, ch["names"][0] + (" ~similar name" if fuzzy else ""), False))
            continue
        words = set(re.findall(r"[a-z]+", name.lower()))
        for g, ws in _GENDER_WORDS.items():
            if words & ws:
                age = next((a for a, aw in _AGE_WORDS.items() if words & aw), None)
                tr = {"gender": g, "age": age, "role": "BACKGROUND", "tags": set()}
                todo.append((3, -len(spans), name, tr, "", True))
                break

    todo.sort(key=lambda t: t[:3])
    used, out = {}, {}
    for _rank, _n, name, tr, matched, via_name in todo:
        best = None
        for alias, (vg, vt) in profiles.items():
            sc = _score(tr, vg, vt, used.get(alias, 0))
            if sc is not None and (best is None or sc > best[0]):
                best = (sc, alias)
        if best:
            used[best[1]] = used.get(best[1], 0) + 1
            out[name] = {"alias": best[1], "reason": _reason(tr, matched, via_name)}
    return out
