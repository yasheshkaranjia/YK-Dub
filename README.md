# YK-Dub — local anime dubbing pipeline

Turns a Japanese-audio anime episode into an English-dubbed one, on CPU
only, using free/open tools — no cloud APIs, no subscriptions. Built
around fansub `.ass`/`.srt` files that already contain a proper human
translation (including on-screen sign/title text), so there's no LLM
translation step to run or wait on.

## Pipeline (4 agents + 2 helpers)

1. **extract_agent.py** — pulls audio out of the video, splits it into a
   **vocals-only** track and an **instrumental** (music/SFX) track with
   Demucs, and grabs the embedded subtitle track if there is one. Writes
   `<name>.manifest.json`.

2. **script_agent.py** — reads the subtitle file directly (no ASR, no
   LLM needed when subs exist):
   - **Dialogue lines** → become the segments that get dubbed, tagged
     with the speaking character's name where the subtitle file has one
     (the ASS "actor" field), for per-character voice assignment.
   - **Sign/title lines** (detected by style name — `Sign`, `OP`, `ED`,
     `Title`, etc. — or a `\pos()`/`\an` position override, the standard
     fansub convention for on-screen text) → pulled into their own
     `signs.ass`, to be burned onto the video later in their original
     position and style, not spoken.
   - If a video has **no subtitle track at all**, this falls back to
     Whisper's own Japanese→English translation for that episode
     (lower quality, but keeps the pipeline running end to end).

3. **dub_agent.py** — synthesizes each dialogue line with Piper TTS
   (using a per-character voice if one's been assigned — see
   `configure_voices.py` below), time-stretches it to fit its subtitle
   window, builds the full vocal track, mixes it with the original
   instrumental track (music/SFX survive), and muxes the result onto
   the video. If there are sign lines, it burns them onto the frames in
   the same step (this forces a full video re-encode for that episode —
   no way around it, since burning text means changing pixels, not just
   copying the stream).

4. **verify_agent.py** — re-extracts the dubbed audio and checks each
   segment's actual speech onset against where it should start, flagging
   anything drifting more than 0.3s, plus an overall duration check.

Two helper scripts sit alongside the pipeline:

- **configure_voices.py** — scans an episode's subtitles for every named
  speaking character and lets you assign each one a Piper voice from
  `voices.json`, saving choices to `voice_map.json`. Remembers earlier
  choices across episodes, so you typically only run this once per
  series (new characters in later episodes still get asked about).
- **check_translation.py** — prints every dialogue line's timing,
  speaker, and text before you commit to a full dub, so you can catch a
  mis-tagged speaker or a sign line that slipped in as dialogue.

Each agent reads/writes a JSON manifest, so any single stage can be
rerun on its own without redoing the others — handy since some steps
are slow.

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
Demucs pulls in PyTorch, so this install is a few GB — expected.

**Whisper model** — only downloads (once, ~500MB) if `script_agent.py`
ever needs its no-subtitle fallback. Cached locally after that.

**Demucs model** — downloads automatically on first use (~80MB), then
runs fully offline. It's a neural net doing the vocal/music split, so
budget a few minutes per episode on CPU.

**Piper TTS voice(s)** — download one or more voices (e.g.
`en_US-lessac-medium`, `en_GB-alan-medium`) from the
[Piper voices page](https://github.com/rhasspy/piper/blob/master/VOICES.md)
— both the `.onnx` and `.onnx.json` files — and place them in the repo
root (or under `piper-voices/`, matching the paths already used in
`voices.json`). **Don't commit these to git** — they're large binaries;
`.gitignore` already excludes `*.onnx` / `*.onnx.json` / `piper-voices/`.

- For a **single voice** for every character, `dub_agent.py`'s
  `PIPER_MODEL` / `PIPER_CONFIG` defaults are enough — no extra setup.
- For **different voices per character**, list each downloaded voice in
  `voices.json` (alias → model/config paths + a label), then run
  `python configure_voices.py "path/to/an/episode.ass"` to assign
  characters to voices interactively. `run.py` also offers to do this
  inline before dubbing each episode.

## 2. Run

**Easiest way — interactive:**
```bash
python run.py
```
It asks where your video (or folder of episodes) is, where to put the
output, and whether you want to review/assign character voices before
dubbing each episode (recommended the first time you dub a new series —
skip it on later episodes of the same series and it'll just reuse
`voice_map.json`). Everything else runs on its own.

**Scripted way — same pipeline, no prompts (good for automation):**
```bash
python orchestrator.py "/path/to/Episode01.mkv" ./work
python orchestrator.py "/path/to/season_folder" ./work   # whole folder, one by one
```

**Setting up character voices manually** (run.py offers this inline,
but you can also run it directly, e.g. to re-pick voices later):
```bash
python configure_voices.py "work/<name>/<name>.ass"
```

Output per episode lands in `./work/<episode name>/`:
- `<name>.dubbed.mp4` — the final dubbed video, music/SFX and translated
  signs intact
- `<name>.translated.json` — every dialogue segment used for dubbing
- `<name>.verification.json` — sync check + any flagged drift

**Resuming a stage** — since each agent reads/writes its own JSON, you
can rerun just one step instead of the whole thing, e.g. after tweaking
`dub_agent.py`:
```bash
python dub_agent.py "work/<name>/<name>.translated.json" "work/<name>/<name>.dubbed.mp4"
```

## 3. Windows-specific gotchas (already fixed in this repo, noted here in case you hit them again)

- **Always activate the venv** before running anything — `(venv)` should
  show in your prompt. If a command isn't found (`piper`, etc.), this is
  almost always why.
- **Subprocess calls use `sys.executable`**, not a bare `"python"` string
  — otherwise Windows can silently run your global Python instead of the
  venv's, and any package versions you installed in the venv won't
  apply to that subprocess.
- **All file reads/writes use `encoding="utf-8"` explicitly** — Windows
  defaults to `cp1252` for text I/O, which crashes on Japanese characters
  in subtitles/manifests.
- **torch/torchaudio/numpy version pinning matters.** Demucs's audio
  saving needs `torch==2.1.0` + `torchaudio==2.1.0` + `numpy<2` + the
  `soundfile` package (torchaudio's actual read/write backend on
  Windows) — mismatches here throw confusing errors like `Numpy is not
  available` or `Couldn't find appropriate backend`.
- **Wrap file paths with spaces/brackets in quotes**, and use
  `Test-Path -LiteralPath "..."` (not plain `Test-Path`) to check a path
  that contains `[` `]` — PowerShell treats brackets as wildcards
  otherwise.

## 4. Low-end laptop tips

- Everything is CPU-only, no GPU required.
- Process a folder overnight; the orchestrator does episodes one at a
  time.
- If Whisper's fallback path is ever too slow, drop `MODEL_SIZE` to
  `"base"` in `script_agent.py`.
- Sanity-check one episode's `.verification.json` before queuing a whole
  season.

## Notes

- `dub_agent.py`'s time-stretch is capped to ffmpeg's single-filter
  `atempo` range (0.5×–2×); a line wildly shorter/longer than its window
  gets clamped rather than sped up or slowed to the point of sounding
  broken.
- A dialogue line that somehow still fails to synthesize (bad text,
  Piper error) is left silent in that spot rather than crashing the
  whole run — check the terminal output for `left it silent` warnings
  after a run to know which lines to spot-check.