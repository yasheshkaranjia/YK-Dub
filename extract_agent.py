"""
Agent 1: Extractor
Pulls audio out of the source video, separates it into a vocals-only
track and an instrumental (music/SFX) track with Demucs, and grabs the
embedded subtitle stream if there is one. Writes a manifest for agent 2.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pysubs2

DEMUCS_MODEL = "htdemucs"  # good quality, CPU-only capable (just slower)
# htdemucs was trained on 7.8s segments - anything longer is a FATAL error
# ("Cannot use a Transformer model with a longer segment than it was
# trained for"), which used to silently kick every GPU run back to CPU.
# --segment only accepts integers, so 7 is the closest safe value.
DEMUCS_GPU_SEGMENT_SEC = "7"  # keeps peak VRAM well under 2GB


def demucs_device_args() -> list:
    """Prefers this machine's GPU if torch can actually see one. Demucs on
    CPU runs close to real-time (roughly as long as the episode itself) -
    on modest laptop hardware that's usually the single biggest time cost
    in the whole pipeline, bigger even than TTS synthesis. Even an
    entry-level GPU is typically several times faster, so it's worth
    using despite limited VRAM - '--segment' caps how much audio Demucs
    processes in one chunk at a time, which is the main lever for staying
    inside a small card's memory (7s keeps peak usage well under 2GB for
    htdemucs; raise it if you have more VRAM to spare, for a small
    speed/memory trade - segment length doesn't affect output quality,
    just how it's chunked internally)."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[extract] using GPU for Demucs: {torch.cuda.get_device_name(0)}")
            return ["-d", "cuda", "--segment", DEMUCS_GPU_SEGMENT_SEC]
    except ImportError:
        pass
    print("[extract] no usable CUDA GPU found for Demucs - running on CPU "
          "(the slow path; expect roughly real-time, so a ~23 min "
          "episode takes ~20-30 min here)")
    return []


def get_duration(video_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return float(out.stdout.strip())


def extract_audio(video_path: str, audio_out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "44100", "-ac", "2", audio_out],
        check=True, capture_output=True, timeout=600,
    )


def separate_vocals(audio_path: str, work_dir: Path, duration_sec: float):
    """Runs Demucs to split into vocals.wav + no_vocals.wav (instrumental).
    PYTHONUTF8=1 is set for this subprocess specifically because Demucs
    internally shells out to ffmpeg and reads its output in text mode -
    on Windows that defaults to cp1252, which crashes on non-ASCII
    output. Setting UTF-8 mode in the child's own environment (this has
    to happen at ITS startup, not ours - a flag can't be applied
    retroactively to an already-running interpreter) fixes it silently,
    with no need to pass `-X utf8` by hand every run.

    A timeout floored at 30 min, scaled to 8x the episode's own runtime,
    covers even a slow CPU-only run comfortably while still being
    BOUNDED - this call used to be able to hang indefinitely (a real
    8+ hour overnight freeze on one episode came from exactly this),
    with no way to tell 'still working' from 'stuck forever' apart."""
    env = {**os.environ, "PYTHONUTF8": "1"}
    device_args = demucs_device_args()
    out_args = ["-o", str(work_dir / "demucs_out"), audio_path]
    gpu_cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", DEMUCS_MODEL,
               *device_args, *out_args]
    cpu_cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", DEMUCS_MODEL, *out_args]
    timeout = max(1800, duration_sec * 8)
    try:
        subprocess.run(gpu_cmd, check=True, env=env, timeout=timeout)
    except subprocess.CalledProcessError:
        if not device_args:
            raise  # already ran on CPU (gpu_cmd == cpu_cmd) - nothing left to fall back to
        print("[extract] GPU run failed (most likely out of VRAM on a small "
              "card) - retrying on CPU instead...")
        subprocess.run(cpu_cmd, check=True, env=env, timeout=timeout)
    stem = Path(audio_path).stem
    sep_dir = work_dir / "demucs_out" / DEMUCS_MODEL / stem
    return sep_dir / "vocals.wav", sep_dir / "no_vocals.wav"


# Language tags that mean "this track is English". ISO 639-2/B ("eng") is
# what ffmpeg/mkv muxers usually write, but hand-muxed releases show up with
# the 639-1 two-letter code, or the full word in a title tag instead, so all
# three spellings are accepted.
ENGLISH_LANGUAGE_TAGS = ("eng", "en", "english")
# Titles that mark a track as signs/songs/forced only. These are the tracks
# that LOOK like a valid English subtitle track but contain almost no spoken
# dialogue - picking one produces a dub with a handful of lines and long
# stretches of silence. Checked against the title tag AND the stream's own
# "forced" disposition, since a forced track is often untitled.
SIGNS_ONLY_HINTS = ("sign", "song", "forced", "karaoke", "op", "ed",
                    "title", "credit", "lyric")
SIGNS_ONLY_EXACT_TOKENS = ("op", "ed")


def probe_subtitle_streams(video_path: str) -> list:
    """Lists every subtitle stream in the container with its language and
    title tags, via one ffprobe call. Returns [] if there are none (or if
    ffprobe fails - the caller treats that the same as "no subtitles", and
    the existing ffmpeg path then reports no track, rather than crashing)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "s",
             "-show_entries",
             "stream=index:stream_tags=language,title:stream_disposition=forced",
             "-of", "json", video_path],
            capture_output=True, text=True, check=True, timeout=60,
        )
        return json.loads(result.stdout).get("streams", [])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, OSError):
        return []


def _stream_language(stream: dict) -> str:
    return (stream.get("tags") or {}).get("language", "").strip().lower()


def _stream_title(stream: dict) -> str:
    return (stream.get("tags") or {}).get("title", "").strip().lower()


def _is_english(stream: dict) -> bool:
    """True if the track's language tag or title says English. A stream
    with NO language tag at all is deliberately not counted as English -
    treating "unknown" as English is how a Japanese track gets picked."""
    language = _stream_language(stream)
    if language in ENGLISH_LANGUAGE_TAGS:
        return True
    # Untagged tracks from some fansub releases put the language only in the
    # title ("English Subs"). Match the language words as whole tokens so a
    # title like "Signs/English" still counts, without "en" matching every
    # title that merely contains those two letters.
    title = _stream_title(stream)
    return any(re.search(rf"\b{re.escape(tag)}\b", title) for tag in ENGLISH_LANGUAGE_TAGS)


def _is_signs_only(stream: dict) -> bool:
    """True if the track is a signs/songs/forced/karaoke track rather than a
    full dialogue track. Short hints (op/ed) are whole-token matched only -
    substring matching would flag a title like "English (Hope subs)" as
    signs, and dropping the real dialogue track over that has already been a
    bug in script_agent.py's style hints."""
    disposition = stream.get("disposition") or {}
    if disposition.get("forced"):
        return True
    title = _stream_title(stream)
    tokens = re.findall(r"[a-z]+", title)
    if any(hint in tokens for hint in SIGNS_ONLY_EXACT_TOKENS):
        return True
    return any(hint in title for hint in SIGNS_ONLY_HINTS
               if hint not in SIGNS_ONLY_EXACT_TOKENS)


def _count_dialogue_lines(stream_path: Path) -> int:
    """Counts real spoken-dialogue lines in an already-extracted subtitle
    file, using the SAME sign-vs-dialogue split script_agent.py dubs with -
    importing it rather than duplicating the rule, so a tiebreak can never
    disagree with what actually ends up in the dub. A track whose lines are
    nearly all signs scores near zero no matter how many events it has."""
    try:
        import script_agent
        subs = pysubs2.load(str(stream_path))
        return sum(1 for e in subs
                   if not e.is_comment
                   and e.plaintext.strip()
                   and not script_agent.is_sign_event(e))
    except Exception:
        # A malformed/unsupported track just scores 0 and loses the tiebreak
        # to a readable one - it must not abort the whole episode.
        return 0


def select_subtitle_stream(video_path: str, work: Path, stem: str) -> tuple:
    """Picks which subtitle stream to dub, returning (stream_index, why).

    Rule, in order:
      1. Only English tracks are candidates at all.
      2. Signs/songs/forced-only tracks are dropped from the candidates.
      3. If that leaves more than one, the one with the most dialogue lines
         wins - that is the "two English tracks, take the one with all the
         character lines" case (e.g. a full-timeline track vs a signs-only
         track that also happens to be tagged English).
      4. If nothing survives (no English at all), fall back to the container's
         first subtitle stream, which is the old hardcoded 0:s:0 behaviour -
         a non-English original still beats no dub at all.

    Returns (None, reason) when the file has no subtitle streams whatsoever.
    """
    streams = probe_subtitle_streams(video_path)
    if not streams:
        return None, "no subtitle streams in the container"

    english = [s for s in streams if _is_english(s)]
    if not english:
        first = streams[0]
        language = _stream_language(first) or "untagged"
        return first.get("index"), (f"no English track found among {len(streams)} "
                                    f"subtitle stream(s) - using stream "
                                    f"{first.get('index')} ({language})")

    dialogue_tracks = [s for s in english if not _is_signs_only(s)]
    if not dialogue_tracks:
        # Every English track looks like signs/songs. Better to use one of
        # them than to fall back to a possibly-Japanese track.
        dialogue_tracks = english

    if len(dialogue_tracks) == 1:
        chosen = dialogue_tracks[0]
        title = _stream_title(chosen) or _stream_language(chosen)
        return chosen.get("index"), f"only English dialogue track ({title})"

    # Genuine tiebreak: count real dialogue lines in each candidate. Each
    # candidate is temporarily extracted as .ass purely to be counted; these
    # are throwaway files rewritten by the real extraction right after.
    scored = []
    for stream in dialogue_tracks:
        probe_out = work / f"{stem}.subprobe{stream.get('index')}.ass"
        try:
            duration = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{stream.get('index')}",
                 "-c:s", "copy", str(probe_out)],
                capture_output=True, text=True, timeout=120,
            )
            if duration.returncode != 0:
                # copy failed (not ASS) - retry as srt, which pysubs2 also reads
                probe_out = work / f"{stem}.subprobe{stream.get('index')}.srt"
                duration = subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{stream.get('index')}",
                     str(probe_out)],
                    capture_output=True, text=True, timeout=120,
                )
            lines = _count_dialogue_lines(probe_out) if duration.returncode == 0 else 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Can't read this candidate - score it 0 and let a readable one
            # win, rather than letting the tiebreak crash the episode.
            lines = 0
        scored.append((lines, stream, probe_out))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_lines, best_stream, _ = scored[0]
    ignored = ", ".join(
        f"stream {s.get('index')} ({_stream_title(s) or _stream_language(s) or 'untitled'}"
        f": {n} lines)" for n, s, _ in scored[1:]
    )

    # The probe files were only needed for counting - leaving them in the
    # work folder would show up as stray subtitles next to the episode's
    # real ones, so they're removed (best-effort: a locked file on Windows
    # shouldn't fail the run over a temp file).
    for _, _, path in scored:
        try:
            path.unlink()
        except OSError:
            pass

    return best_stream.get("index"), (
        f"{len(scored)} English tracks - chose stream {best_stream.get('index')} "
        f"with most dialogue lines ({best_lines}); ignored {ignored}"
    )


def extract_subtitles(video_path: str, ass_out: str, srt_out: str,
                      work: Path = None, stem: str = None) -> str:
    """Tries to copy the subtitle stream exactly as .ass (keeps styles,
    position tags, and the actor/name field intact). Falls back to a
    plain .srt conversion only if the source track isn't ASS/SSA.

    Uses select_subtitle_stream() to choose WHICH stream, instead of the
    first one in the container."""
    if work is not None and stem is not None:
        stream_index, reason = select_subtitle_stream(video_path, work, stem)
    else:
        stream_index, reason = 0, "no work dir given - defaulting to stream 0:s:0"

    if stream_index is None:
        print(f"[extract] {reason}")
        return None

    print(f"[extract] subtitle track: {reason}")
    stream_map = f"0:{stream_index}"

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-map", stream_map, "-c:s", "copy", ass_out],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and Path(ass_out).exists():
            return ass_out

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-map", stream_map, srt_out],
            capture_output=True, text=True, timeout=120,
        )
        return srt_out if result.returncode == 0 and Path(srt_out).exists() else None
    except FileNotFoundError:
        # ffmpeg missing from PATH. Previously this raised out of run() and
        # killed the whole episode; returning None instead lets script_agent
        # fall back to its Whisper translation, so one missing tool degrades
        # to a lower-quality dub rather than no output at all.
        print("[extract] ffmpeg not found on PATH - cannot read embedded "
              "subtitles (script_agent will fall back to Whisper)")
        return None
    except subprocess.TimeoutExpired:
        print("[extract] reading the subtitle track timed out - skipping subs")
        return None


def run(video_path: str, work_dir: str, external_srt: str = None) -> dict:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    audio_path = work / f"{stem}.wav"
    duration_sec = get_duration(video_path)  # computed once, reused below instead of a 2nd ffprobe call

    extract_audio(video_path, str(audio_path))

    print("[extract] separating vocals from music/SFX with Demucs "
          "(slow the first time - it downloads a model, then a few minutes per episode)...")
    vocals_path, instrumental_path = separate_vocals(str(audio_path), work, duration_sec)

    if external_srt:
        sub_path = Path(external_srt)
    else:
        found = extract_subtitles(video_path, str(work / f"{stem}.ass"),
                                  str(work / f"{stem}.srt"), work, stem)
        sub_path = Path(found) if found else None

    manifest = {
        "video_path": str(Path(video_path).resolve()),
        "audio_path": str(audio_path.resolve()),
        "vocals_path": str(vocals_path.resolve()),
        "instrumental_path": str(instrumental_path.resolve()),
        "subtitle_path": str(sub_path.resolve()) if sub_path else None,
        "duration_sec": duration_sec,
    }
    manifest_path = work / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[extract] {stem}: audio split into vocals/instrumental + "
          f"{'subs' if manifest['subtitle_path'] else 'no subs found'} -> {manifest_path}")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_agent.py <video_path> <work_dir> [external_srt]")
        sys.exit(1)
    video, work_dir = sys.argv[1], sys.argv[2]
    srt = sys.argv[3] if len(sys.argv) > 3 else None
    run(video, work_dir, srt)

