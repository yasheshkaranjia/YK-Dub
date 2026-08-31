# Local anime dubbing pipeline (4 agents)

Turns a Japanese-audio video into an English-dubbed one, on CPU only,
using free/open tools. Runs one episode at a time, so it's fine on a
low-end laptop overnight.

## Pipeline

1. **extract_agent.py** — pulls a 16kHz mono WAV and the embedded
   subtitle track (if any) out of the video, writes `<name>.manifest.json`.
2. **translate_agent.py** — runs Whisper's `translate` task on the audio
   (Japanese speech → English text) and compares each line to the
   existing subtitle text. Flags lines where they disagree so you can
   spot-check just those, instead of everything.
3. **dub_agent.py** — synthesizes each line with Piper TTS, time-stretches
   it to fit its subtitle window (so it lands on the right timestamp),
   builds the full audio track, and muxes it onto the original video
   (video stream copied, not re-encoded — fast).
4. **verify_agent.py** — re-extracts the dubbed audio and checks each
   segment's actual speech onset against where it was supposed to start,
   flagging anything that drifted more than 0.3s, plus an overall
   duration check.

Each agent reads/writes a JSON file, so you can rerun any single stage
without redoing the others, or inspect the manifests as you go.

## 1. Install (one-time)

**System packages**
```bash
sudo apt update
sudo apt install ffmpeg python3-venv python3-pip
```

**Python environment**
```bash
cd dub_pipeline
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Whisper model** — nothing to do manually; the first run of
`translate_agent.py` downloads the `small` model (~500MB) once and caches
it locally, so every run after that is fully offline. If your laptop
struggles, open `translate_agent.py` and change `MODEL_SIZE = "small"`
to `"base"` (smaller, faster, slightly less accurate).

**Piper TTS voice** — Piper needs a voice model + config, not just the
`pip install`. Download a voice (e.g. `en_US-lessac-medium`) from the
Piper voices release page on GitHub, and place both the `.onnx` and
`.onnx.json` files in the `dub_pipeline` folder — or edit `PIPER_MODEL` /
`PIPER_CONFIG` in `dub_agent.py` to point wherever you saved them.

## 2. Run

Single episode:
```bash
python orchestrator.py "/path/to/Episode01.mkv" ./work
```

Whole folder (processed one by one, in filename order):
```bash
python orchestrator.py "/path/to/season_folder" ./work
```

Output per episode lands in `./work/<episode name>/`:
- `<name>.dubbed.mp4` — the final dubbed video
- `<name>.translated.json` — every line + flag for mismatches
- `<name>.verification.json` — sync check + any flagged drift

## 3. Low-end laptop tips

- Everything here is CPU-only — no GPU required, and nothing needs a
  network connection except the one-time model downloads.
- Process the folder overnight rather than in parallel; the orchestrator
  already does episodes one at a time.
- If `translate_agent.py` is too slow, drop to Whisper's `base` model.
- If you want to sanity-check a run before committing the whole season,
  run the orchestrator on one episode first and check its
  `.verification.json` (`segments_flagged` should be a small fraction of
  `segments_checked`) before queuing the rest.

## Notes

- If a video has no embedded subtitle track, `extract_agent.py` sets
  `subtitle_path` to `null` and `translate_agent.py` just uses Whisper's
  own translation for every line (flagged `no_subtitle_match` so you know
  which lines had no subtitle to cross-check against).
- `dub_agent.py`'s time-stretch is capped to ffmpeg's single-filter
  `atempo` range (0.5×–2×). A line that's wildly shorter/longer than its
  subtitle window will still be clamped to that range rather than sped up
  or slowed down to the point of sounding broken.
