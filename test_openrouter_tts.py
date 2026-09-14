"""
Standalone smoke test for the OpenRouter/Fish Audio voice-cloning
integration - isolated from the full dub_agent.py pipeline, so a bad API
key, missing reference clip, or network issue shows up in one quick call
instead of partway through a full episode run.

Usage (venv activated, from project root):
    python test_openrouter_tts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dub_agent import synth_segment_openrouter  # noqa: E402


def main():
    voice_cfg = {"reference_clip": "reference_clip.wav"}
    test_line = "Weak, you're too weak! Can you at least make this a little interesting?"
    out_path = "openrouter_test_output.wav"

    print(f"Reference clip: {voice_cfg['reference_clip']}")
    print(f"Text: {test_line!r}")
    print("Calling OpenRouter...")

    try:
        synth_segment_openrouter(test_line, out_path, voice_cfg)
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"\nSUCCESS - wrote {out_path} ({size_kb:.1f} KB)")
    print("Play it back and judge whether it sounds like LIAM's voice.")


if __name__ == "__main__":
    main()
