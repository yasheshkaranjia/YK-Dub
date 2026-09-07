"""
Agent 1: Extractor
Pulls audio out of the source video, separates it into a vocals-only
track and an instrumental (music/SFX) track with Demucs, and grabs the
embedded subtitle stream if there is one. Writes a manifest for agent 2.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

DEMUCS_MODEL = "htdemucs"  # good quality, CPU-only capable (just slower)
DEMUCS_GPU_SEGMENT_SEC = "8"  # keeps peak VRAM well under 2GB - see demucs_device_args()


def demucs_device_args() -> list:
    """Prefers this machine's GPU if torch can actually see one. Demucs on
    CPU runs close to real-time (roughly as long as the episode itself) -
    on modest laptop hardware that's usually the single biggest time cost
    in the whole pipeline, bigger even than TTS synthesis. Even an
    entry-level GPU is typically several times faster, so it's worth
    using despite limited VRAM - '--segment' caps how much audio Demucs
    processes in one chunk at a time, which is the main lever for staying
    inside a small card's memory (8s keeps peak usage well under 2GB for
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


def extract_subtitles(video_path: str, ass_out: str, srt_out: str) -> str:
    """Tries to copy the subtitle stream exactly as .ass (keeps styles,
    position tags, and the actor/name field intact). Falls back to a
    plain .srt conversion only if the source track isn't ASS/SSA."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", "-c:s", "copy", ass_out],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0 and Path(ass_out).exists():
        return ass_out

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", srt_out],
        capture_output=True, text=True, timeout=120,
    )
    return srt_out if result.returncode == 0 and Path(srt_out).exists() else None


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
        found = extract_subtitles(video_path, str(work / f"{stem}.ass"), str(work / f"{stem}.srt"))
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

