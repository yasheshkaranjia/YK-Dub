# Local anime dubbing pipeline (4 agents)

Turns a Japanese-audio video into an English-dubbed one, on CPU only,
using free/open tools. Runs one episode at a time, so it's fine on a
low-end laptop overnight.

## Pipeline

1. **extract_agent.py** — pulls audio out of the video, splits it into a
   **vocals-only** track and an **instrumental** (music/SFX) track with
   Demucs, and grabs the embedded subtitle track (if any). Writes
   `<name>.manifest.json`.
2. **translate_agent.py** — transcribes the isolated Japanese vocals with
   Whisper (with timestamps), then translates each line with a local LLM
   via **Ollama**, giving it a couple of lines of surrounding context so
   it reads like a conversation, and asking it to reword the line to fit
   naturally in the clip's original duration. Cross-checks against the
   existing subtitle text and flags lines worth a second look. If Ollama
   isn't running, it falls back to Whisper's own translation (no context,
   no reworking) so the pipeline still works.
3. **dub_agent.py** — synthesizes each line with Piper TTS, time-stretches
   it to fit its subtitle window, builds the full vocal track, **mixes it
   with the original instrumental track** from step 1 (so music/SFX
   survive), and muxes the result onto the video (video stream copied,
   not re-encoded).
4. **verify_agent.py** — re-extracts the dubbed audio and checks each
   segment's actual speech onset against where it should start, flagging
   anything drifting more than 0.3s, plus an overall duration check.

Each agent reads/writes a JSON file, so you can rerun any single stage
without redoing the others.

## 1. Install (one-time)

**System packages**
```bash
sudo apt update
sudo apt install ffmpeg python3-venv python3-pip
```
(Windows: `winget install ffmpeg`. Mac: `brew install ffmpeg`.)

**Python environment**
```bash
cd YK-Dub
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Demucs pulls in PyTorch, so this install is noticeably bigger than
before — that's expected.

**Whisper model** — nothing to do manually; the first run of
`translate_agent.py` downloads the `small` model (~500MB) once and caches
it locally. If your laptop struggles, change `MODEL_SIZE = "small"` to
`"base"` in `translate_agent.py`.

**Demucs model** — also downloads automatically on first use (~80MB),
then runs fully offline. It's a neural net doing the vocal/music split,
so expect it to take a few minutes per episode on CPU — it's the slowest
step to add, but it's what lets the music and sound effects survive.

**Ollama (for real translation)**
```bash
# install from https://ollama.com, then:
ollama pull qwen3:4b
```
Leave the Ollama app/service running before you start the pipeline —
`translate_agent.py` talks to it over `localhost:11434`. If you skip
this or forget to start it, the pipeline still runs, just falls back to
Whisper's blunter built-in translation with no context or timing
rewrite. Got 8GB+ RAM to spare? `ollama pull qwen3:8b` instead for
noticeably better translations; tight on RAM? `qwen3:1.7b`.

**Piper TTS voice** — download a voice (e.g. `en_US-lessac-medium`) from
the Piper voices page on GitHub — both the `.onnx` and `.onnx.json` files
— and place them in the repo root, or point `PIPER_MODEL` / `PIPER_CONFIG`
in `dub_agent.py` at wherever you saved them.

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
- `<name>.dubbed.mp4` — the final dubbed video, music/SFX intact
- `<name>.translated.json` — every line (Japanese + final English) + flag
- `<name>.verification.json` — sync check + any flagged drift

## 3. Low-end laptop tips

- Everything is CPU-only, no GPU required — the vocal/music split and
  translation model both run locally.
- Process a folder overnight; the orchestrator already does episodes one
  at a time.
- If Whisper is too slow, drop `MODEL_SIZE` to `"base"`.
- If Ollama translation feels slow, drop to `qwen3:1.7b`.
- Sanity-check one episode's `.verification.json` before queuing a whole
  season.

## Notes

- If a video has no embedded subtitle track, lines are flagged
  `no_subtitle_match` — there's simply nothing to cross-check against,
  the translation itself is unaffected.
- `dub_agent.py`'s time-stretch is capped to ffmpeg's single-filter
  `atempo` range (0.5×–2×); a line wildly shorter/longer than its window
  gets clamped rather than sped up or slowed to the point of sounding
  broken.
- The similarity check in `translate_agent.py` is a rough consistency
  check between the final English line and the existing subtitle, not a
  meaning-level check — a `mismatch` flag doesn't always mean the
  translation is wrong, just that it reads differently from the subtitle.
