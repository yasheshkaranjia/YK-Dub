# YK-Dub — local anime dubbing pipeline

Turns a Japanese-audio anime episode into an English-dubbed one, on CPU
only, using free/open tools — no cloud APIs, no subscriptions. Built
around the official English `.ass` subtitles that ship with the
release (dialogue AND on-screen sign/title text), with a different
Piper TTS voice assignable per character, the original Japanese audio
and English subtitles kept as switchable tracks, and background
music/SFX preserved underneath the dub.

This README covers everything: how it works, how to set it up from
scratch, and every real bug we hit and fixed along the way — read this
once and you shouldn't need to rediscover any of it.

---

## How it works

**run.py** is what you actually run. It's interactive: asks for a
video or folder path, a work/output folder, and whether to review
character voices before dubbing — then chains the 4 agents below,
per episode, one at a time.

1. **extract_agent.py**
   - Pulls audio out of the video, splits it into a **vocals-only**
     track and an **instrumental** (music/SFX) track using Demucs.
   - Grabs the subtitle track in its **original `.ass` format**
     (`-c:s copy`, no conversion) — this matters because `.ass` carries
     the Actor/speaker-name field, styles, and position tags that
     `.srt` simply doesn't have. Falls back to `.srt` only if the
     source track isn't ASS/SSA.
   - Writes `<name>.manifest.json`.

2. **script_agent.py** — reads the subtitle directly, no ASR/LLM
   needed when subs exist:
   - **Dialogue lines** become the segments that get dubbed, tagged
     with the speaking character's name for voice assignment.
   - **Sign/title lines** are detected by style name (`Sign`, `OP`,
     `ED`, `Title`) OR by the Actor field (some releases tag signs
     there instead — e.g. `SIGN`, `EPTITLE`, `NEXTEPTITLE`) OR by a
     `\pos()`/`\an` position override. These are pulled into their own
     `signs.ass`, burned onto the video later in their original
     position/style — not spoken.
   - If a video has **no subtitle track at all**, falls back to
     Whisper's own Japanese-to-English translation for that episode
     only (lower quality, no speaker names, but keeps the pipeline
     running).

3. **dub_agent.py** — the most complex stage:
   - Synthesizes each dialogue line with Piper TTS, using a different
     voice per character if configured (see Voice Setup below).
   - **Text cleanup before synthesis**: stutters written as `A-All` or
     `O-Oh` get turned into `A... All` (a real hesitation pause instead
     of Piper trying to pronounce a broken word), and honorifics like
     `Liam-sama` become `Liam sama` (same idea — the hyphen was being
     read literally).
   - **Natural pacing over brute-force stretching**: every line first
     synthesizes at normal pace, then checks how far off it is from its
     subtitle window. If a line would need extreme time-stretching
     (outside roughly 0.75x-1.4x), it's re-synthesized once more using
     Piper's own `--length-scale` (a genuine pace change, like a person
     talking faster or slower) to get close to the target duration
     naturally — only the small remaining gap gets handled by ffmpeg's
     `atempo` filter, instead of atempo doing all the work and sounding
     robotic.
   - Mixes the finished vocal track with the original instrumental
     (music/SFX survive), using **raw numpy sample arrays** rather than
     repeated `pydub.overlay()` calls — overlay() copies the entire
     track's audio on every single call, which made a long episode with
     hundreds of lines take dramatically longer than it should (a real,
     confirmed bug — fixed).
   - Muxes the result onto the video with:
     - the English dub as the default audio track
     - the original Japanese audio kept as a second, switchable track
     - the original English subtitles embedded as a switchable
       subtitle track (`mov_text`, mp4's native format)
     - sign text burned onto the frames, if any exist — this forces a
       full video re-encode (no way around it, burning text means
       changing pixels, not copying a stream). Forced to 8-bit
       (`-pix_fmt yuv420p`) even when the source is 10-bit HEVC —
       10-bit x264 encoding is dramatically slower for no visible
       benefit in a dub, and this alone was the difference between a
       roughly 12-minute re-encode and one that took over 12 hours on
       a 10-bit source before this fix.
     - When burning is happening, ffmpeg's own live progress prints to
       the terminal (deliberately not captured/hidden) — this used to
       look identical to a hang with no way to tell it was still
       working.
   - **Preflight voice check**: before synthesis starts, every voice
     file a given episode actually needs is checked to exist on disk.
     A missing `.onnx`/`.onnx.json` used to only surface as one silent
     failure per affected line, discovered only after a full run.
   - A line that still fails to synthesize for any other reason is left
     silent in that spot rather than crashing the whole run.

4. **verify_agent.py** — checks each segment's actual speech onset
   against where it should start, plus an overall duration check.
   Checks the pre-mix, speech-only vocal track, not the final
   music-mixed video — checking the final video was a real bug that
   caused a near-universal false "drift" flag on 80-95% of lines in
   every episode, because background music playing continuously meant
   silence-detection almost never found real silence to measure from.

**configure_voices.py** scans a subtitle for every speaking character
(skips signs automatically) and lets you assign each one a Piper voice
from `voices.json`, interactively. Saves to `voice_map.json`, which
`dub_agent.py` reads automatically. Remembers earlier choices.

**collect_dubbed.py** gathers every episode's `*.dubbed.mp4` out of its
own subfolder into one flat folder, once a whole season is done, so you
can copy the whole thing to a phone or USB drive in one go.

---

## 1. Install (one-time)

**System packages**
```
sudo apt update
sudo apt install ffmpeg python3-venv python3-pip
```
(Windows: `winget install ffmpeg`. Mac: `brew install ffmpeg`.)

**Python environment**
```
git clone https://github.com/yasheshkaranjia/YK-Dub.git
cd YK-Dub
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(Windows activation: `venv\Scripts\activate`)

Demucs pulls in PyTorch, so this install is a few GB — expected.

**Whisper model** only downloads (once, about 500MB) if `script_agent.py`
ever needs its no-subtitle fallback.

**Demucs model** downloads automatically on first use (about 80MB),
then runs fully offline. Budget a few minutes per episode on CPU for
the actual vocal/music split — this is one of the two slowest steps.

### Piper TTS voices

None of the voice model files are committed to this repo — large
binaries, deliberately kept out by `.gitignore`. On a fresh clone:

**Base voices** — download from the Piper voices page
(github.com/rhasspy/piper/blob/master/VOICES.md), place both the
`.onnx` and `.onnx.json` in the repo root:
- `en_US-lessac-medium` (default voice)
- `en_US-amy-medium` (a female US voice)

**More variety** — clone the full voice repo into the project root:
```
git clone https://huggingface.co/rhasspy/piper-voices
```
Not every voice ships in every quality — some only have a `low`
version, not `medium` (this bit us once: `southern_english_female`
only exists in `low`, and `voices.json` pointed at a `medium` path
that never existed, causing every line using it to fail silently).
Check what actually exists before adding a voice to `voices.json`:
```
Get-ChildItem "piper-voices\en\en_GB" -Recurse -Filter "*.onnx"
```
(Mac/Linux: `find piper-voices/en/en_GB -name "*.onnx"`)

`voices.json` in this repo lists every voice tested so far with its
exact path and a human-readable label — add an entry the same way for
anything new.

Multi-speaker models (`vctk`, and partly `aru`/`semaine`) aren't fully
supported — `dub_agent.py` always uses speaker index 0, so they'll work
but won't let you pick a specific speaker inside the file.

---

## 2. Assigning character voices

```
python configure_voices.py <path-to-a-subtitle.ass>
```
Lists your available voices, then walks through every real speaking
character found (skips `SIGN`/`EPTITLE`/etc. automatically), asking one
at a time — type a number to assign a voice, or press Enter to leave it
on the default. Saved to `voice_map.json`.

When picking voices, double-check character genders rather than
guessing from an unfamiliar name — worth a quick search for the actual
character list before assigning. We initially mixed up genders (e.g.
assigned a male antagonist a female voice) purely from unfamiliar name
spellings.

---

## 3. Run

```
python run.py
```
Asks for a video/folder path, a work folder, and whether to review
voices per episode. Output per episode lands in
`<work_folder>/<episode name>/`:
- `<name>.dubbed.mp4` — final dubbed video
- `<name>.translated.json` — every dialogue segment used for dubbing
- `<name>.verification.json` — sync check plus any flagged drift

Non-interactive / scripted use is still supported:
```
python run.py "/path/to/Episode01.mkv" ./work
```

Resuming a single stage — since each agent reads and writes its own
JSON, you can rerun just one step instead of the whole thing:
```
python dub_agent.py "work/<name>/<name>.translated.json" "work/<name>/<name>.dubbed.mp4"
python verify_agent.py "work/<name>/<name>.dubbed.json"
```

After a whole season is done:
```
python collect_dubbed.py ./work
```
Copies every `*.dubbed.mp4` into one flat `./collected` folder.

---

## 4. Realistic time expectations (measured, not guessed)

Per roughly 22-24 minute episode, on a CPU-only laptop:
- Demucs extraction: about 20-29 minutes
- Piper synthesis: about 15-27 minutes (a bit more for episodes with
  several outlier-tempo lines needing the two-pass length-scale
  correction)
- Final mux: about 1-2 minutes for plain episodes; about 15 minutes for
  episodes with sign lines, since those need a video re-encode (this
  step took 6 to over 12 hours before the 8-bit fix below — if a
  burn-in step is taking dramatically longer than 15-20 minutes,
  something has regressed)
- Verify: under a minute

A full 12-episode season: roughly 9-11 hours, run back to back.

Before a long run: pause Windows Update (Settings, Windows Update,
Pause updates) and disable sleep while plugged in. An auto-restart
mid-run cost an entire 12-hour re-encode once.

---

## 5. Before trusting a downloaded episode file

Corrupted or truncated downloads can silently produce a dubbed video
that's missing the back half of the episode, with no obvious error.
Check first:
```
ffmpeg -i "your_episode.mkv" -map 0:a:0 -f null -
```
The final `time=` line should reach the file's real runtime with no
`EBML` or `File ended prematurely` errors. If it stops short, the file
itself is bad — re-download rather than debugging the pipeline.

---

## 6. Windows-specific gotchas (already fixed in this repo)

- Always activate the venv — `(venv)` should show in your prompt.
- UTF-8 everywhere: Windows defaults to `cp1252` for text I/O, which
  crashes on Japanese characters in subtitles and manifests. Every file
  read/write in this repo explicitly uses `encoding="utf-8"`. Demucs's
  own internal subprocess calls also get `PYTHONUTF8=1` set in their
  environment automatically — no need to pass `-X utf8` yourself.
- Subprocess calls use `sys.executable`, not a bare `"python"` string —
  otherwise Windows can silently run global Python instead of the
  venv's, and packages installed in the venv won't apply.
- torch/torchaudio/numpy version pinning matters. Demucs's audio saving
  needs `torch==2.1.0` plus `torchaudio==2.1.0` plus `numpy<2` plus the
  `soundfile` package (torchaudio's actual read/write backend on
  Windows) — mismatches throw errors like `Numpy is not available` or
  `Couldn't find appropriate backend`.
- Quote paths with spaces or brackets, and use
  `Test-Path -LiteralPath "..."` rather than plain `Test-Path` for a
  path containing `[` or `]` — PowerShell treats brackets as wildcards.
- PowerShell doesn't support `&&` — use `;` or separate lines.
- `.gitignore` must match your actual output folder name. If you run
  `run.py` and give it a folder name other than `work` (for example
  `final`), and `.gitignore` only excludes `work/`, everything in that
  folder — including large `.dubbed.mp4` files — will get swept into
  `git add .`. This repo's `.gitignore` excludes `work/`, `final/`,
  `*.mp4`, and `*.mkv` to cover this.

---

## 7. Optional: run on Google Colab (free GPU)

`YK_Dub_Colab.ipynb` runs the same pipeline with a free GPU, which
mainly speeds up the Demucs step. Upload it to
colab.research.google.com, set Runtime, Change runtime type, GPU, and
follow the cells in order — videos and voice files need to live on
Google Drive first, since Colab has no persistent local disk. Piper
synthesis itself won't get much faster there — it's already
CPU-optimized.

---

## Notes

- `dub_agent.py`'s remaining `atempo` correction is capped to ffmpeg's
  single-filter range (0.5x-2x); combined with the length-scale
  pre-correction above, this should rarely be hit hard in practice now.
- A dialogue line that fails to synthesize for any reason prints a
  `left it silent` or `failed to synthesize` warning and is skipped
  rather than crashing the whole run — check the terminal output after
  a run to know which lines to spot-check.
- If a whole voice consistently fails across many lines, check its
  `model`/`config` paths in `voices.json` actually exist on disk — the
  preflight check at the start of `dub_agent.py` should now catch this
  immediately instead of letting a run finish with silent gaps.
