"""
Agent 1: Extractor
Pulls audio out of the source video, separates it into a vocals-only
track and an instrumental (music/SFX) track with Demucs, and grabs the
embedded subtitle stream if there is one. Writes a manifest for agent 2.
"""
import json
import subprocess
import sys
from pathlib import Path

DEMUCS_MODEL = "htdemucs"  # good quality, CPU-only capable (just slower)


def get_duration(video_path: str) -> float:
    # ffprobe, not ffmpeg - it's built for reading metadata and doesn't
    # need to decode any actual frames, so this returns almost instantly
    # even on a huge file.
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def extract_audio(video_path: str, audio_out: str) -> None:
    # -vn drops the video stream entirely (we only want audio here).
    # 44.1kHz stereo matches what Demucs' pretrained model expects as
    # input - feeding it something else technically works but risks
    # subtly worse separation.
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "44100", "-ac", "2", audio_out],
        check=True, capture_output=True,
    )


def separate_vocals(audio_path: str, work_dir: Path):
    """Runs Demucs to split into vocals.wav + no_vocals.wav (instrumental)."""
    # sys.executable (not just "python") makes sure Demucs runs inside
    # THIS venv's Python, not whatever "python" happens to resolve to on
    # the system PATH - avoids a whole category of "it's not installed"
    # confusion when multiple Python versions are on the machine.
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", DEMUCS_MODEL,
         "-o", str(work_dir / "demucs_out"), audio_path],
        check=True,
    )
    # Demucs names its own output folder after the input file's stem and
    # the model used - this has to match that convention exactly or the
    # files "exist" but at a path we're not looking at.
    stem = Path(audio_path).stem
    sep_dir = work_dir / "demucs_out" / DEMUCS_MODEL / stem
    return sep_dir / "vocals.wav", sep_dir / "no_vocals.wav"


def extract_subtitles(video_path: str, ass_out: str, srt_out: str) -> str:
    """Tries to copy the subtitle stream exactly as .ass (keeps styles,
    position tags, and the actor/name field intact). Falls back to a
    plain .srt conversion only if the source track isn't ASS/SSA."""
    # -c:s copy (not a re-encode) preserves the file byte-for-byte - this
    # matters a lot here specifically, because script_agent.py depends on
    # the style names and \pos() tags surviving intact to tell dialogue
    # from signs. A lossy conversion could silently break that detection.
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", "-c:s", "copy", ass_out],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and Path(ass_out).exists():
        return ass_out

    # Source track wasn't ASS/SSA (e.g. it was already SRT, or a bitmap
    # format ffmpeg can transcode) - .srt has no style/position info at
    # all, so sign detection degrades to text-pattern matching only.
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", srt_out],
        capture_output=True, text=True,
    )
    return srt_out if result.returncode == 0 and Path(srt_out).exists() else None


def run(video_path: str, work_dir: str, external_srt: str = None) -> dict:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    audio_path = work / f"{stem}.wav"

    extract_audio(video_path, str(audio_path))

    print("[extract] separating vocals from music/SFX with Demucs "
          "(slow the first time - it downloads a model, then a few minutes per episode)...")
    vocals_path, instrumental_path = separate_vocals(str(audio_path), work)

    # external_srt lets you hand-supply a subtitle file (e.g. you found a
    # better translation than what's muxed into the video) instead of
    # whatever ffmpeg pulls out automatically.
    if external_srt:
        sub_path = Path(external_srt)
    else:
        found = extract_subtitles(video_path, str(work / f"{stem}.ass"), str(work / f"{stem}.srt"))
        sub_path = Path(found) if found else None

    # Every path in here gets .resolve()'d to absolute - later agents run
    # from whatever directory the user happens to be in, and a relative
    # path that was valid from the original working directory can quietly
    # point nowhere from a different one.
    manifest = {
        "video_path": str(Path(video_path).resolve()),
        "audio_path": str(audio_path.resolve()),
        "vocals_path": str(vocals_path.resolve()),
        "instrumental_path": str(instrumental_path.resolve()),
        "subtitle_path": str(sub_path.resolve()) if sub_path else None,
        "duration_sec": get_duration(video_path),
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

