# YK-Dub — local anime dubbing pipeline

Turns a Japanese-audio anime episode into an English-dubbed one, on CPU
only, using free/open tools - no cloud APIs, no subscriptions. Built
around fansub `.ass` files that already contain a proper human
translation (dialogue AND on-screen sign/title text), with different
Piper TTS voices assignable per character.

## Pipeline (4 agents + a setup helper)

1. **extract_agent.py** — pulls audio out of the video, splits it into a
   **vocals-only** track and an **instrumental** (music/SFX) track with
   Demucs, and grabs the embedded subtitle track in its original `.ass`
   format (falls back to `.srt` only if the source track isn't ASS/SSA -
   note `.srt` has no speaker-name field, so per-character voices won't
   work for those). Writes `<name>.manifest.json`.

2. **script_agent.py** — reads the subtitle file directly (no ASR, no
   LLM needed when subs exist):
   - **Dialogue lines** → become the segments that get dubbed, tagged
     with the speaking character's name (from the subtitle's Actor
     field) for voice assignment.
   - **Sign/title lines** (detected by style name - `Sign`, `OP`, `ED`,
     `Title` - OR by the Actor field, since some fansub groups tag signs
     that way instead - e.g. `SIGN`, `EPTITLE` - OR by a `\pos()`/`\an`
     position override) → pulled into their own `signs.ass`, burned onto
     the video later in their original position/style, not spoken.
   - If a video has **no subtitle track at all**, falls back to
     Whisper's own Japanese→English translation (lower quality, no
     speaker names, but keeps the pipeline running).

3. **dub_agent.py** — synthesizes each dialogue line with Piper TTS,
   using a different voice per character if one's been configured (see
   Voice Setup below), time-stretches it to fit its subtitle window,
   builds the full vocal track, mixes it with the original instrumental
   (music/SFX survive), and muxes the result onto the video with:
   - the English dub as the default audio track
   - the **original Japanese audio kept as a second, switchable track**
   - the original English subtitles embedded as a switchable subtitle track
   - sign text burned onto the frames, if any were found (this one part
     forces a full video re-encode - no way around it, burning text
     means changing pixels, not copying a stream)

4. **verify_agent.py** — re-extracts the dubbed audio and checks each
   segment's actual speech onset against where it should start, flagging
   anything drifting more than 0.3s, plus an overall duration check.

**configure_voices.py** — interactive helper: scans a subtitle file for
every speaking character (skips signs automatically) and asks you to
assign each one a Piper voice from `voices.json`. Saves to
`voice_map.json`, which `dub_agent.py` reads automatically. Remembers
earlier choices, so re-running it on a new episode only asks about names
it hasn't seen yet.

**run.py** — the actual thing you run. Interactive: asks for a video or
folder path, whether to review character voices before dubbing, then
chains all 4 agents per episode automatically.

## 1. Install (one-time)

**System packages**
```bash
sudo apt update
sudo apt install ffmpeg python3-venv python3-pip
```
(Windows: `winget install ffmpeg`. Mac: `brew install ffmpeg`.)

**Python environment**
```bash
git clone https://github.com/yasheshkaranjia/YK-Dub.git
cd YK-Dub
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Demucs pulls in PyTorch, so this is a few GB - expected.

**Whisper model** — only downloads (once, ~500MB) if `script_agent.py`
ever needs its no-subtitle fallback. Cached locally after that.

**Demucs model** — downloads automatically on first use (~80MB), then
runs fully offline. Budget a few minutes per episode on CPU for the
actual vocal/music split.

**Ollama** - not needed anymore. An earlier version of this pipeline used
Ollama + an LLM for translation before it was rebuilt around reading
fansub subtitles directly. If you still have it installed, it's safe to
ignore/remove.

### Piper TTS voices

None of the voice model files are committed to this repo - they're large
binaries and `.gitignore` deliberately keeps them out. On a fresh clone
you need to fetch them yourself:

**Base voices** (put these two `.onnx` + `.onnx.json` pairs directly in
the repo root) - download from the
[Piper voices page](https://github.com/rhasspy/piper/blob/master/VOICES.md):
- `en_US-lessac-medium` (default voice)
- `en_US-amy-medium` (a female US voice)

**More voice variety** - clone the whole `piper-voices` model repo into
the project root for a much bigger selection (GB accents, more female
voices, etc.):
```bash
git clone https://huggingface.co/rhasspy/piper-voices
```
This creates a `piper-voices/` folder in the repo root, already excluded
by `.gitignore`. **Not every voice ships in every quality** - some only
have a `low` version, not `medium` - check what actually exists before
adding it to `voices.json`:
```bash
Get-ChildItem "piper-voices\en\en_GB" -Recurse -Filter "*.onnx"   # Windows
find piper-voices/en/en_GB -name "*.onnx"                          # Mac/Linux
```

**voices.json** already in this repo lists the voices this project has
been tested with and their exact paths - if you add more voices from the
repo, add an entry here too (`model`, `config`, and a human-readable
`label`) so `configure_voices.py` can offer them.

Multi-speaker models (`vctk`, and to an extent `aru`/`semaine`) aren't
fully supported yet - `dub_agent.py` always uses speaker index 0 from
whichever model it's given, so a multi-speaker file will work but won't
let you pick a specific speaker inside it.

## 2. Run

```bash
python run.py
```

It'll ask for:
1. **A video file OR a folder of episodes** - paste the path, quotes are
   fine either way.
2. **Work/output folder** - defaults to `./work`.
3. **Whether to review character voices** before dubbing each episode -
   say yes the first time you dub a new series; it'll list every
   speaking character found in that episode's subtitles and let you pick
   a voice per name (or press Enter to leave everyone on the default).

Output per episode lands in `<work_folder>/<episode name>/`:
- `<name>.dubbed.mp4` — the final dubbed video
- `<name>.translated.json` — every dialogue segment used for dubbing,
  including which character said each line
- `<name>.verification.json` — sync check + any flagged drift

**Resuming a single stage** — since each agent reads/writes its own
JSON, you can rerun just one step instead of the whole thing, e.g. after
tweaking `voice_map.json`:
```bash
python dub_agent.py "work/<name>/<name>.translated.json" "work/<name>/<name>.dubbed.mp4"
```

**Non-interactive / scripted use** is still supported:
```bash
python run.py "/path/to/Episode01.mkv" ./work
```

## 3. Windows-specific gotchas (already fixed in this repo, noted here in case you hit them again)

- **Always activate the venv** before running anything - `(venv)` should
  show in your prompt. If a command isn't found (`piper`, etc.), this is
  almost always why.
- **UTF-8 issues** — Windows defaults to `cp1252` for text I/O, which
  breaks on Japanese characters in subtitles/manifests and on some of
  Demucs's own internal output. This repo already sets `PYTHONUTF8=1`
  for the Demucs subprocess specifically, and uses `encoding="utf-8"`
  explicitly on every file read/write - you shouldn't need to pass
  `-X utf8` yourself anymore.
- **Subprocess calls use `sys.executable`**, not a bare `"python"` string
  - otherwise Windows can silently run your global Python instead of the
  venv's, and any package versions you installed in the venv won't
  apply to that subprocess.
- **torch/torchaudio/numpy version pinning matters.** Demucs's audio
  saving needs `torch==2.1.0` + `torchaudio==2.1.0` + `numpy<2` + the
  `soundfile` package (torchaudio's actual read/write backend on
  Windows) - mismatches here throw confusing errors like `Numpy is not
  available` or `Couldn't find appropriate backend`.
- **Wrap file paths with spaces/brackets in quotes**, and use
  `Test-Path -LiteralPath "..."` (not plain `Test-Path`) to check a path
  that contains `[` `]` - PowerShell treats brackets as wildcards
  otherwise.
- **PowerShell doesn't support `&&`** to chain commands like bash does -
  use `;` instead, or just separate lines.

## 4. Before trusting a downloaded episode file

Corrupted/truncated downloads happen, and they can silently produce a
dubbed video that's missing the back half of the episode with no error
anywhere obvious. Worth a quick check before running the full pipeline
on a new file:
```bash
ffmpeg -i "your_episode.mkv" -map 0:a:0 -f null -
```
Look at the final `time=` line - it should reach the file's real
runtime with no `EBML`/`File ended prematurely` errors. If it stops
short, the file itself is bad - re-download it rather than debugging the
pipeline.

## 5. Low-end laptop tips

- Everything is CPU-only, no GPU required.
- Process a folder overnight; `run.py` does episodes one at a time.
- If Whisper's fallback path is ever too slow, drop `MODEL_SIZE` to
  `"base"` in `script_agent.py`.
- Sanity-check one episode's `.verification.json` before queuing a whole
  season.

## Notes

- `dub_agent.py`'s time-stretch is capped to ffmpeg's single-filter
  `atempo` range (0.5×-2×); a line wildly shorter/longer than its window
  gets clamped rather than sped up or slowed to the point of sounding
  broken.
- A dialogue line that fails to synthesize (bad text, missing/broken
  voice file) is left silent in that spot rather than crashing the whole
  run - check the terminal output for `failed to synthesize` or
  `left it silent` warnings after a run to know which lines to
  spot-check. If a whole voice consistently fails, double-check its
  `model`/`config` paths in `voices.json` actually point at files that
  exist on disk.
