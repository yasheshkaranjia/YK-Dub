"""
Agent 4: Verifier
Confirms the dubbed video's runtime matches the source, then checks each
segment's actual speech onset (via silence detection) against where the
timestamp said it should start, flagging anything that drifted too far.
Nothing here requires reading the dialogue - it only compares numbers.
"""
import json
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

DRIFT_THRESHOLD_SEC = 0.3
WINDOW_SEC = 1.5  # search window around each expected start time


def get_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def extract_audio(video_path: str, out_wav: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_wav],
        check=True, capture_output=True,
    )


def actual_onset(audio: AudioSegment, expected_start: float):
    window_start_ms = max(0, int((expected_start - WINDOW_SEC / 2) * 1000))
    window_end_ms = min(len(audio), int((expected_start + WINDOW_SEC / 2) * 1000))
    clip = audio[window_start_ms:window_end_ms]
    if len(clip) == 0:
        return None
    spans = detect_nonsilent(clip, min_silence_len=100, silence_thresh=clip.dBFS - 16)
    if not spans:
        return None
    return window_start_ms / 1000.0 + spans[0][0] / 1000.0


def run(dubbed_manifest_path: str) -> dict:
    data = json.loads(Path(dubbed_manifest_path).read_text(encoding="utf-8"))
    work_dir = Path(dubbed_manifest_path).parent / "verify_work"
    work_dir.mkdir(exist_ok=True)
    check_wav = work_dir / "check.wav"

    extract_audio(data["dubbed_video"], str(check_wav))
    audio = AudioSegment.from_wav(check_wav)

    src_dur = data["duration_sec"]
    dub_dur = get_duration(data["dubbed_video"])
    duration_ok = abs(src_dur - dub_dur) < 0.5

    report = []
    for seg in data["segments"]:
        onset = actual_onset(audio, seg["start"])
        drift = None if onset is None else round(onset - seg["start"], 3)
        report.append({
            "start": seg["start"], "end": seg["end"],
            "detected_onset": onset, "drift_sec": drift,
            "flagged": drift is not None and abs(drift) > DRIFT_THRESHOLD_SEC,
        })

    flagged_count = sum(1 for r in report if r["flagged"])
    result = {
        "dubbed_video": data["dubbed_video"],
        "duration_match": duration_ok,
        "source_duration": src_dur,
        "dubbed_duration": dub_dur,
        "segments_checked": len(report),
        "segments_flagged": flagged_count,
        "details": report,
    }
    out_path = Path(str(dubbed_manifest_path).replace(".dubbed.json", ".verification.json"))
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[verify] duration match: {duration_ok} | {flagged_count}/{len(report)} "
          f"segments flagged (drift > {DRIFT_THRESHOLD_SEC}s) -> {out_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_agent.py <dubbed_manifest.json>")
        sys.exit(1)
    run(sys.argv[1])