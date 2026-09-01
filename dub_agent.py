"""
Agent 3: Synthesizer + Muxer
Turns each translated line into English speech with Piper TTS, time-
stretches every clip to fit its subtitle window so it lands where the
original timestamp says it should, assembles a full-length audio track,
and muxes it onto the source video. The video stream is copied (not
re-encoded), so this step stays fast even on modest hardware.
"""
import json
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm

# Point these at whatever Piper voice you downloaded (see README).
PIPER_MODEL = "en_US-lessac-medium.onnx"
PIPER_CONFIG = "en_US-lessac-medium.onnx.json"


def synth_segment(text: str, out_wav: str) -> None:
    subprocess.run(
        ["piper", "--model", PIPER_MODEL, "--config", PIPER_CONFIG,
         "--output_file", out_wav],
        input=text, text=True, encoding="utf-8", check=True, capture_output=True,
    )


def stretch_to_duration(in_wav: str, out_wav: str, target_sec: float) -> None:
    current = AudioSegment.from_wav(in_wav).duration_seconds
    if current <= 0:
        Path(in_wav).rename(out_wav)
        return
    tempo = max(0.5, min(2.0, current / target_sec))  # ffmpeg atempo's safe range
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_wav, "-filter:a", f"atempo={tempo}", out_wav],
        check=True, capture_output=True,
    )


def build_vocal_track(segments: list, work_dir: Path, total_duration: float) -> Path:
    track = AudioSegment.silent(duration=int(total_duration * 1000))
    skipped = 0
    for i, seg in enumerate(tqdm(segments, desc="[dub] synthesizing lines", unit="line")):
        # A line where translation fell back to the raw source text (e.g. an
        # Ollama timeout) is still Japanese - Piper's English voice can't
        # speak it, so leave that window silent instead of crashing.
        if seg.get("japanese_text") and seg["final_text"] == seg["japanese_text"]:
            tqdm.write(f"[dub] segment {i} was never translated - leaving it silent")
            skipped += 1
            continue

        raw = work_dir / f"seg_{i:04d}_raw.wav"
        fitted = work_dir / f"seg_{i:04d}_fit.wav"
        try:
            synth_segment(seg["final_text"], str(raw))
            stretch_to_duration(str(raw), str(fitted), max(seg["end"] - seg["start"], 0.3))
        except subprocess.CalledProcessError as e:
            tqdm.write(f"[dub] segment {i} failed to synthesize ({e}) - leaving it silent")
            skipped += 1
            continue

        clip = AudioSegment.from_wav(fitted)
        track = track.overlay(clip, position=int(seg["start"] * 1000))

    if skipped:
        print(f"[dub] {skipped}/{len(segments)} lines left silent (untranslated or synth failure)")
    out_path = work_dir / "dubbed_vocals.wav"
    track.export(out_path, format="wav")
    return out_path


def mix_with_instrumental(vocals_path: Path, instrumental_path: str, work_dir: Path) -> Path:
    out_path = work_dir / "final_mix.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(vocals_path), "-i", instrumental_path,
         "-filter_complex", "amix=inputs=2:duration=longest:dropout_transition=0",
         str(out_path)],
        check=True, capture_output=True,
    )
    return out_path


def escape_for_ffmpeg_filter(path: str) -> str:
    """ffmpeg filter syntax treats ':' and '\\' specially - this makes a
    Windows path safe to drop into a subtitles= filter argument."""
    return path.replace("\\", "/").replace(":", "\\:")


def mux(video_path: str, audio_path: Path, out_path: str, signs_path: str = None) -> None:
    if signs_path:
        # Burning text onto frames means the video must be re-encoded -
        # a plain stream copy can't add pixels to existing frames.
        filt = f"subtitles='{escape_for_ffmpeg_filter(signs_path)}'"
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", str(audio_path),
             "-filter_complex", f"[0:v]{filt}[v]",
             "-map", "[v]", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", "-shortest", out_path],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", str(audio_path),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
            check=True, capture_output=True,
        )


def run(translated_manifest_path: str, out_video: str) -> dict:
    data = json.loads(Path(translated_manifest_path).read_text(encoding="utf-8"))
    work_dir = Path(translated_manifest_path).parent / "tts_work"
    work_dir.mkdir(exist_ok=True)

    track = build_vocal_track(data["segments"], work_dir, data["duration_sec"])
    final_mix = mix_with_instrumental(track, data["instrumental_path"], work_dir)
    mux(data["video_path"], final_mix, out_video, data.get("signs_path"))

    result = {**data, "dubbed_video": str(Path(out_video).resolve())}
    result_path = Path(str(translated_manifest_path).replace(".translated.json", ".dubbed.json"))
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dub] wrote {out_video} -> manifest at {result_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dub_agent.py <translated_manifest.json> <out_video.mp4>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
