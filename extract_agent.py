"""
Agent 1: Extractor
Pulls a mono 16kHz WAV (for ASR) and the first subtitle stream (SRT) out of
a source video, writing a JSON manifest that agent 2 picks up next.
"""
import json
import subprocess
import sys
from pathlib import Path


def get_duration(video_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def extract_audio(video_path: str, audio_out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", audio_out],
        check=True, capture_output=True,
    )


def extract_subtitles(video_path: str, sub_out: str) -> bool:
    """Returns True if an embedded subtitle track was found and pulled out."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", sub_out],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and Path(sub_out).exists()


def run(video_path: str, work_dir: str, external_srt: str = None) -> dict:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    audio_path = work / f"{stem}.wav"
    sub_path = work / f"{stem}.srt"

    extract_audio(video_path, str(audio_path))

    if external_srt:
        sub_path = Path(external_srt)
    elif not extract_subtitles(video_path, str(sub_path)):
        sub_path = None  # agent 2 will fall back to pure ASR translation

    manifest = {
        "video_path": str(Path(video_path).resolve()),
        "audio_path": str(audio_path.resolve()),
        "subtitle_path": str(sub_path.resolve()) if sub_path else None,
        "duration_sec": get_duration(video_path),
    }
    manifest_path = work / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[extract] {stem}: audio + "
          f"{'subs' if manifest['subtitle_path'] else 'no subs found'} -> {manifest_path}")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_agent.py <video_path> <work_dir> [external_srt]")
        sys.exit(1)
    video, work_dir = sys.argv[1], sys.argv[2]
    srt = sys.argv[3] if len(sys.argv) > 3 else None
    run(video, work_dir, srt)