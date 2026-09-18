"""Replace a dubbed MP4's English audio with a corrected WAV."""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python replace_dub_audio.py <dubbed.mp4> <final_mix.wav> <output.mp4>")
        sys.exit(1)

    video, audio, output = map(Path, sys.argv[1:])
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    if not audio.exists():
        raise SystemExit(f"WAV not found: {audio}")
    if output.resolve() == video.resolve():
        raise SystemExit("Output must be a different file from the input video.")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
            "-map", "0:v:0",       # existing video
            "-map", "1:a:0",       # corrected English mix
            "-map", "0:a:1?",      # existing Japanese track, if present
            "-map", "0:s?",        # existing subtitles, if present
            "-c:v", "copy",
            "-c:a:0", "aac", "-b:a:0", "192k",
            "-c:a:1", "copy",
            "-c:s", "copy",
            "-metadata:s:a:0", "language=eng",
            "-disposition:a:0", "default",
            "-disposition:a:1", "0",
            str(output),
        ],
        check=True,
    )
    print(f"Created: {output}")


if __name__ == "__main__":
    main()
