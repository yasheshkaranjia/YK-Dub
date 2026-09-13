"""
Benchmarks VoxCPM2 (via llama.cpp-omni's CPU/GPU GGUF path) on THIS
machine, before any of it gets wired into the dubbing pipeline.

Why this exists: VoxCPM2's own published RTF numbers (~0.30 on an
RTX 4090, ~1.76 on Apple M4 Pro) are from hardware nothing like an
i5-1135G7 + MX350 2GB - the M4 Pro number in particular is Metal-GPU-
accelerated, not CPU-only, so it isn't a fair stand-in either. The only
way to know if this is fast enough for real batch dubbing on THIS
laptop is to actually run it here and measure.

RTF (real-time factor) = wall-clock synthesis time / output audio
duration. RTF 1.0 means "takes exactly as long as the clip is." Piper
on this machine's CPU is well under 1.0 per line. Anything meaningfully
above ~2-3 probably isn't going to be practical for a ~300-line episode
even with parallel workers - see the verdict this script prints.

--- One-time setup (do this before running this script) ---

1. Build llama.cpp-omni (needs a C++ toolchain - on Windows, install
   "Desktop development with C++" via the Visual Studio Build Tools
   installer first, plus CMake):

       git clone https://github.com/tc-mb/llama.cpp-omni.git
       cd llama.cpp-omni
       cmake -B build -DCMAKE_BUILD_TYPE=Release
       cmake --build build --target voxcpm2-cli -j
       # -> produces build/bin/voxcpm2-cli(.exe)

2. Download the GGUF weights (BaseLM + Acoustic) from:
       https://huggingface.co/DennisHuang648/VoxCPM2-GGUF
   Q8_0 for the BaseLM is a good default (half the download of F16,
   the model card states negligible quality loss).

3. Get one short (5-15s) clean reference clip of a character's voice
   to clone - you already have these for free, courtesy of Demucs:
   pull one from any episode's tts_work or manifest vocals.wav for a
   character who has a clear, mostly-solo line.

Usage:
    python benchmark_voxcpm2.py <path_to_voxcpm2-cli> <BaseLM.gguf> <Acoustic.gguf> [reference_clip.wav]

If reference_clip.wav is omitted, this benchmarks plain (non-cloned)
synthesis only - still useful for a raw speed number, just not
testing the cloning path this project actually wants to use.
"""
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# A handful of lines spanning realistic anime-dialogue lengths - short
# reactive lines and longer expository ones - so the RTF reported isn't
# skewed by testing only one length of line.
TEST_LINES = [
    "Wait!",
    "I don't understand what you're saying.",
    "This is exactly the kind of thing I was warning you about earlier.",
    "You really think that's going to work? After everything that's happened?",
    "No matter what happens next, I'm not backing down from this fight.",
]


def wav_duration_sec(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def run_one(cli_path: str, base_lm: str, acoustic: str, text: str,
            reference_clip: str, out_wav: str) -> dict:
    cmd = [cli_path, "-t", text, "-o", out_wav]
    if reference_clip:
        cmd += ["-r", reference_clip]
    cmd += [base_lm, acoustic]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    elapsed = time.time() - start

    if result.returncode != 0 or not Path(out_wav).exists():
        return {"text": text, "ok": False, "stderr": result.stderr[-500:]}

    duration = wav_duration_sec(out_wav)
    rtf = elapsed / duration if duration > 0 else None
    return {"text": text, "ok": True, "elapsed_sec": elapsed,
            "audio_sec": duration, "rtf": rtf}


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    cli_path, base_lm, acoustic = sys.argv[1], sys.argv[2], sys.argv[3]
    reference_clip = sys.argv[4] if len(sys.argv) > 4 else None

    if not Path(cli_path).exists():
        print(f"Can't find the built binary at {cli_path} - did the cmake build finish? "
              f"See the setup steps in this script's docstring.")
        sys.exit(1)
    if not Path(base_lm).exists() or not Path(acoustic).exists():
        print(f"Can't find the GGUF weight files. Checked:\n  {base_lm}\n  {acoustic}")
        sys.exit(1)

    print(f"Testing {'cloned voice' if reference_clip else 'default voice (no cloning)'} "
          f"synthesis across {len(TEST_LINES)} lines of varying length...\n")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(TEST_LINES):
            out_wav = str(Path(tmp) / f"test_{i}.wav")
            r = run_one(cli_path, base_lm, acoustic, text, reference_clip, out_wav)
            results.append(r)
            if r["ok"]:
                print(f"  \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
                print(f"    {r['elapsed_sec']:.2f}s wall time for {r['audio_sec']:.2f}s of audio "
                      f"-> RTF {r['rtf']:.2f}")
            else:
                print(f"  FAILED: \"{text[:50]}\" - {r['stderr']}")

    ok_results = [r for r in results if r["ok"]]
    if not ok_results:
        print("\nEvery test line failed - check the stderr output above before benchmarking further.")
        sys.exit(1)

    avg_rtf = sum(r["rtf"] for r in ok_results) / len(ok_results)
    print(f"\n{'=' * 60}")
    print(f"Average RTF across {len(ok_results)}/{len(TEST_LINES)} successful lines: {avg_rtf:.2f}")
    print(f"{'=' * 60}")

    # Rough, honest framing - not a hard cutoff, just a sanity check
    # against what this project's actual workload looks like: a ~23min
    # episode has roughly 300-400 lines. Piper currently handles that in
    # ~13-28 min with 4 parallel workers.
    if avg_rtf <= 1.5:
        print("\nVERDICT: Fast enough to be genuinely usable for batch dubbing, likely close to "
              "or better than the current Piper step once run through the same parallel-worker "
              "setup dub_agent.py already uses for Piper. Worth building the real integration.")
    elif avg_rtf <= 4.0:
        print("\nVERDICT: Workable but slow - an overnight batch run is realistic, a same-day "
              "turnaround per episode probably isn't without parallelizing across several "
              "workers (same thread-pool approach as Piper). Worth trying on ONE character "
              "before committing the whole cast to it.")
    else:
        print("\nVERDICT: Too slow to be practical for this project's batch-dubbing use case on "
              "this hardware. Worth trying the RVC-on-Kokoro path instead (see the project guide) "
              "rather than pushing further on VoxCPM2 here.")


if __name__ == "__main__":
    main()
