# YK-Dub — local anime dubbing pipeline

Turns a Japanese-audio anime episode into an English-dubbed one, running
locally on modest hardware, using free/open tools — no subscriptions. Built
around the official English `.ass` subtitles that ship with the
release (dialogue AND on-screen sign/title text), with a different
TTS voice assignable per character (Supertonic 3 by default — see Voice
Setup below — with Kokoro/Piper still available as legacy engines), the
original Japanese audio
and English subtitles kept as switchable tracks, and background
music/SFX preserved underneath the dub.

Point it at a release with a dozen subtitle languages and it picks the
right one by itself, assigns a distinct voice to every named character,
and can produce either a self-contained file or a **lossless** one that
keeps the original video untouched (see "Why the output is smaller than
the source", which also explains how to avoid ever re-encoding the video).

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
   - **Chooses WHICH subtitle track to use** instead of blindly taking
     the first one (see "Subtitle track selection" below) — a real
     release carries a dozen language tracks, and the old hardcoded
     `-map 0:s:0` picked whichever the muxer happened to write first.
   - Grabs the chosen subtitle track in its **original `.ass` format**
     (`-c:s copy`, no conversion) — this matters because `.ass` carries
     the Actor/speaker-name field, styles, and position tags that
     `.srt` simply doesn't have. Falls back to `.srt` only if the
     source track isn't ASS/SSA.
   - Writes `<name>.manifest.json`.

### Subtitle track selection (`extract_agent.py`)

Real WEB-DL releases ship **one subtitle stream per language** — the
episode this was built against has **14** (Arabic, German, Spanish x2,
French, Indonesian, Italian, Portuguese, Russian, Thai, Vietnamese,
Chinese x2, English). `select_subtitle_stream()` picks one, in this order:

1. **Only English tracks are candidates.** Recognised from the stream's
   `language` tag (`eng`/`en`) or its `title` ("English Subs"). A track
   with **no** language tag is deliberately *not* treated as English —
   that is exactly how a Japanese track gets picked.
2. **Signs/songs/forced-only tracks are dropped.** These look like a
   valid English subtitle but contain almost no spoken dialogue, so
   picking one yields a dub with a handful of lines and long silences.
   Detected from the title, and from the stream's own `forced`
   disposition. Short hints (`op`/`ed`) are matched as **whole tokens
   only** — substring matching reads a title like "Hope subs" as signs
   and throws away the real dialogue track.
3. **If several English tracks survive, the one with the most dialogue
   lines wins.** Each candidate is temporarily extracted and counted with
   the *same* sign-vs-dialogue rule `script_agent.py` dubs with, so the
   count can never disagree with what actually ends up in the dub.
4. **Nothing English found → falls back to the first subtitle stream**
   (the old behaviour), with a printed note. A non-English original still
   beats no dub at all.

The choice is always printed, e.g.
`subtitle track: 2 English tracks - chose stream 3 with most dialogue lines (334); ignored stream 2 (Signs: 0 lines)`,
so a wrong pick is visible during the run rather than after watching it.

2. **script_agent.py** — reads the subtitle directly, no ASR/LLM
   needed when subs exist:
   - **Dialogue lines** become the segments that get dubbed, tagged
     with the speaking character's name for voice assignment, and
     flagged as internal monologue if the source line is italicized
     (`\i1`) — the usual fansub convention for a character's unspoken
     thoughts — so `dub_agent.py` can read it softer/quieter instead of
     identically to spoken lines.
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
     of Piper trying to pronounce a broken word), honorifics like
     `Liam-sama` become `Liam sama` (same idea — the hyphen was being
     read literally), and ALL-CAPS shouted words (`STOP!`) get
     Title-cased before synthesis so Piper's phonemizer doesn't mistake
     them for acronyms and spell them out letter by letter.
   - **Tone-aware delivery**: every line is read for cheap textual cues —
     ALL CAPS or `!!!` (shouting), a trailing `!` or an unambiguously
     celebratory phrase such as “happy birthday” (excitement), `...` or
     a trailing `-` (hesitation/trailing off), a trailing `?` (a
     question), and whether the line is italicized in the source `.ass`
     (fansub convention for internal monologue). Each cue nudges that
     line's Piper `--noise-scale`/`--noise-w` (vocal variation),
     `--length-scale` (pace), `--sentence-silence` (pause length), and a
     post-synthesis volume trim. Excited, shouted, and questioning lines
     also receive a small formant-preserving pitch lift, which gives the
     text-only Supertonic/Kokoro engines a more perceptible delivery change
     without changing the character's identity. Piper settings are applied
     *relative to that specific voice's own tuned defaults* (read from its
     `.onnx.json`), not one flat setting for every voice and every line.
     This is a rule-based approximation, not real emotional TTS, but it's
     enough to stop every line from being read in an identical flat tone.
     Tune the multipliers in
     `classify_tone()` in `dub_agent.py` to taste; the run prints a
     summary of how many lines got each tag (e.g.
     `tone-adjusted delivery applied - exclaim: 42, hesitant: 11, shout: 3`).
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
   - **Known separation limitation:** Demucs sometimes puts foley that
     overlaps dialogue (such as clothing movement) in its vocals stem, so
     that effect is lost when Japanese speech is removed. Mixing any of the
     stem back also leaks spatially positioned Japanese speakers, especially
     in multi-character scenes. The pipeline therefore prioritizes clean
     English dialogue and does not recover that stem. A future fix needs a
     stronger dialogue-versus-foley separation model or editable production
     stems rather than stereo/center filtering.
   - The mix itself runs at **48 kHz stereo with the music ducked under
     speech** — an earlier version mixed at the TTS engine's own 22-24 kHz
     mono rate, which dragged the music down with it (everything above
     ~12 kHz discarded, stereo collapsed to mono) and played the BGM at
     full volume under every line. Now: both tracks upsample to 48 kHz
     stereo first, the instrumental is sidechain-ducked while a line is
     spoken and swells back between lines, and the whole mix is
     loudness-normalized to -16 LUFS / -1.5 dBTP (EBU R128) so every
     episode comes out at the same consistent volume. The dub's AAC track
     encodes at 192 kb/s as a result — higher quality than the original
     Japanese track kept alongside it.
   - Lines that already fit their subtitle window keep their **natural
     pace** — a line finishing early is padded with silence instead of
     being stretched to fill the window (a 2s read in a 4s window used to
     come out at half speed). Speed-ups for overrunning lines are
     unchanged.
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
   - **`mux()` has two very different paths and only one is lossless.**
     With no signs, the video is stream-copied (`-c:v copy`) and is
     bit-for-bit the source. With signs, the whole episode is decoded
     and re-encoded (see "Why the output is smaller than the source"
     below). The quality target for that re-encode is
     `DEFAULT_VIDEO_CRF` (16), overridable without editing code via the
     `YKDUB_VIDEO_CRF` environment variable.
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

**`voice_map.json` is cumulative and matched case-insensitively**
(`normalize_speaker()` upper-cases both sides), so the same show's later
episodes reuse every earlier assignment automatically. Running
dub_agent without an entry for a speaker is not silent: the run prints a
`WARNING - these speaker(s) have NO entry in voice_map.json` block with
the line count for each one, **before** the synthesis, so a whole
episode coming out in one voice is caught up front instead of after
watching it. A quick pre-assignment check for one episode:
```python
# every speaker in the episode's English subtitle, and whether it is mapped
import pysubs2, script_agent, json
subs = pysubs2.load("probe.ass")
names = {(e.name or "").strip() for e in subs
         if not e.is_comment and e.plaintext.strip()
         and not script_agent.is_sign_event(e)}
mapped = {k.upper() for k in json.load(open("voice_map.json")) if not k.startswith("_")}
print("unmapped:", sorted(n for n in names if n and n.upper() not in mapped))
```
Beware **compound actor names** — some releases write `Dylan/Zoey` or
`Jessie/Anna/Evelyn` for a shared line. These match nothing unless given
their own explicit entry, so add one (it takes the voice of the character
speaking first).

**collect_dubbed.py** gathers every episode's `*.dubbed.mp4` out of its
own subfolder into one flat folder, once a whole season is done, so you
can copy the whole thing to a phone or USB drive in one go.

**work/final_pack.py** (created during a run) builds the **lossless**
final file for an episode: the original video stream-copied untouched,
plus the new English dub, the original Japanese audio, and only the
English subtitle track. See "Why the output is smaller than the source"
above — it exists because the normal mux re-encodes the video whenever
there is sign text to burn in. It reads the episode's `.dubbed.json`,
so it only works after the dub stage has finished.

---

## Why the output is smaller than the source (and how to avoid it)

An episode can come out **far smaller** than the file it started from —
we hit 386 MB from a 1431 MB source. The audio was the only thing meant
to change, so this looks like something went wrong. It is worth
understanding exactly why, because the answer is not obvious and two
different causes can stack.

### 1. Burning signs forces a full video re-encode

`dub_agent.mux()` has **two** paths, and only one of them is lossless:

| Condition | What happens to the video |
| --- | --- |
| No sign text | `-c:v copy` — **bit-for-bit identical**, fast |
| Sign text present | every frame decoded, text composited, whole video re-encoded |

A **soft subtitle is a stream** the player draws over untouched frames.
A **burned-in subtitle is pixels** — you cannot copy frames that don't
contain text into frames that do. So flattening sign text into the image
is a destructive operation on the video by definition.

The measured result on a real episode: source video **7.99 Mbps**, output
**1.88 Mbps** — a **4.25x** drop, because the burn-in ran at CRF 20 and
anime compresses very efficiently at flat colours. The source author's
8 Mbps was generous for this content, so it was not visibly destroyed,
but it *was* a genuine quality reduction.

**Fixes applied:**
- `DEFAULT_VIDEO_CRF` raised 20 → **16**, and exposed as the
  `YKDUB_VIDEO_CRF` environment variable (validated — a bad value falls
  back to the default with a warning rather than failing the mux after
  the episode has already been synthesized).
- libx264 preset `fast` → `medium`. The burn-in is dominated by the
  subtitle compositing, not the encoder, so the slower preset buys real
  compression efficiency for a small share of the total time.
- The run now prints the CRF it is using, so the quality target is
  visible instead of implicit.

### 2. Dropped subtitle languages and font attachments

The mux keeps only the English subtitle stream, so a 14-language release
loses 13 subtitle tracks and its embedded fonts. Small next to the video,
but non-zero.

### The better approach: keep the original video, add only the audio

Because audio and video are independent streams, you can add a dub
**without touching the video at all**:

```
ffmpeg -i original.mkv -i dub_mix.wav \
  -map 0:v:0 -c:v copy \
  -map 1:a:0 -map 0:a:0 \
  -c:a:0 aac -b:a:0 192k -c:a:1 copy \
  -metadata:s:a:0 language=eng -disposition:a:0 default \
  -metadata:s:a:1 language=jpn -disposition:a:1 0 \
  -map 0:s:2 -c:s copy \
  output.mkv
```

This gives **zero video loss**, keeps all the original subtitle tracks and
fonts you choose to map, and takes seconds instead of ~5 minutes because
nothing is re-encoded. `work/final_pack.py` in this repo does exactly
this (video copied, English dub added, Japanese kept, English subs kept,
other languages dropped) — verified with an MD5 of the raw video stream,
which matched the source exactly.

**The one trade-off:** signs become a *toggleable soft subtitle* rather
than being always visible. With burned-in signs they show even with
subtitles off; as a soft track, turning subtitles off also hides them.

### 3. A silent bug this exposed: lost `PlayResX`/`PlayResY`

This is the most important lesson in this README, because it produced a
**visibly broken** frame that no amount of extra bitrate could fix, and it
was only caught by actually looking at the output.

The symptom: sign text landed in the wrong place and **overlapped
itself** — "Fire"/"Water"/"Wind" and a column of letters all stacked on
top of each other. It looked exactly like severe compression damage.

**The actual cause:** ASS positioning is **resolution-relative**, not
absolute pixels. `\pos(168, 102)` means "168/PlayResX across, 102/PlayResY
down". The source subtitle declares `PlayResX: 640 / PlayResY: 360`, so
that point is 26% across and 28% down.

`script_agent.split_subtitles()` built its `signs.ass` from fresh
`pysubs2.SSAFile()` objects and copied `subs.styles` — but **`PlayResX`
and `PlayResY` live in `.info`, not `.styles`**, and `.info` was left
empty. Without them, libass falls back to its own default of **384x288**,
so every `\pos()` was measured against the wrong coordinate space and
landed in the wrong place, rescaled, overlapping.

There was already a comment warning that the sign file would "lose its
font/color/position info" if styles weren't copied — but it only
addressed styles, and **PlayRes is the coordinate space those positions
are relative to.**

**Fix:** copy `.info` across too, falling back to the source's own values:
```python
dialogue.info = dict(subs.info)
signs.info = dict(subs.info)
for f in (dialogue, signs):
    f.info.setdefault("PlayResX", "640")
    f.info.setdefault("PlayResY", "360")
```

**If you see signs in the wrong place or overlapping:** check that the
generated `signs.ass` has `PlayResX`/`PlayResY` in its `[Script Info]`
section, and that they match the source subtitle. Missing = wrong
layout, guaranteed.

---

## Running a long dub unattended (Windows)

Two problems bite when a full episode run (30-60 minutes) is started from
a terminal:

1. **The run dies when its shell window closes.** Launching the pipeline
   as a child of an interactive shell means the process is killed with
   that shell. It must be detached — a Windows Scheduled Task works.
2. **`forrtl: error (200): program aborting due to window-CLOSE event`.**
   The Intel Fortran runtime inside torch/MKL (used by Demucs) aborts
   when its console disappears, which is exactly what a windowless
   scheduled task causes. Setting these in the runner fixes it:
   ```
   set KMP_DUPLICATE_LIB_OK=TRUE
   set OMP_NUM_THREADS=4
   set MKL_NUM_THREADS=4
   set PYTHONUNBUFFERED=1
   set PYTHONIOENCODING=utf-8
   ```

**Hardware encoding is also lost under a hidden task.** A run that logged
`software libx264 (no usable hardware encoder found)` was doing the same
job its normal shell did with `h264_qsv` — measured at **5.0x realtime
vs 0.53x**, i.e. ~4 minutes instead of ~45 for one episode. If a burn-in
step is far slower than the timings in section 4 suggest, check which
encoder the run actually reported.

**Piping answers into the interactive `run.py`:** PowerShell prepends a
BOM to piped input, which corrupts the first line (the path becomes
`\ufeffC:\...` and `Path.exists()` fails). Write the answers to a file
with a BOM-less encoding and redirect it in:
```powershell
[System.IO.File]::WriteAllLines("$PWD\answers.txt", $lines,
    (New-Object System.Text.UTF8Encoding $false))
```
then `venv\Scripts\python.exe run.py < answers.txt`. For unattended runs,
`orchestrator.py` is the better entry point anyway — it is fully
non-interactive.

**A locked output file fails the mux with `Permission denied`.** If a
player (VLC, Windows Films & TV) has the previous output open, ffmpeg
cannot overwrite it and the run fails at the very last step. Close the
player before re-running.

---

**check_translation.py** prints each dialogue line's timing, speaker,
and final text from a `.translated.json` — a quick way to catch a
speaker name that didn't get picked up, or a sign line that slipped
through as dialogue, before committing to a full dub run:
```
python check_translation.py "work/<name>/<name>.translated.json" [--speaker NAME]
```

**trim_translated.py** cuts a `.translated.json` (and its referenced
audio) down to just the first N seconds, so you can dub and listen to a
short test clip of a new voice/tone setting instead of waiting for a
full 20+ minute episode run.

**orchestrator.py** + **watchdog.py** — `orchestrator.py` runs the full
pipeline non-interactively over one video or a whole folder, one
episode at a time, and is resume-safe at both the episode and stage
level (see its own docstring). `watchdog.py` relaunches it after a
crash for unattended overnight batches — `run.py` is the interactive
entry point most people want; reach for these two only for scripted or
overnight runs.

---

## 1. Install (one-time)

**System packages**
```
sudo apt update
sudo apt install ffmpeg python3-venv python3-pip
```
(Windows: `winget install ffmpeg`. Mac: `brew install ffmpeg`.)

**Windows: verify `ffmpeg` is actually on PATH** after installing. A
winget install can succeed without adding its `bin` folder to PATH, and
the pipeline then fails at the first extract step. Confirm with
`Get-Command ffmpeg`; if it is missing, add the winget package's `bin`
folder to your user PATH:
```powershell
$bin = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" `
        -Recurse -Filter ffmpeg.exe | Select-Object -First 1).DirectoryName
[Environment]::SetEnvironmentVariable("Path", "$bin;" + `
    [Environment]::GetEnvironmentVariable("Path","User"), "User")
```
(Open a new terminal afterwards.)

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

### Piper TTS voices (legacy — model files not installed)

**This setup now defaults to the Supertonic engine** — its model downloads
automatically and needs no manual voice files (see "Supertonic engine"
below). Piper still works as an engine, but the `piper-voices/` model
folder was removed to keep the working directory light. To use a Piper
voice again, re-download its `.onnx` + `.onnx.json` from
github.com/rhasspy/piper/blob/master/VOICES.md into the matching path and
assign it in `voice_map.json` as before.

The instructions below are kept for that case.

### Piper TTS voices

None of the voice model files are committed to this repo — large
binaries, deliberately kept out by `.gitignore`. On a fresh clone:

**Base voices** — download from the Piper voices page
(github.com/rhasspy/piper/blob/master/VOICES.md), place both the
`.onnx` and `.onnx.json` in the matching directory under
`piper-voices/en/en_US/<voice>/<quality>/`:
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

### Optional: Kokoro TTS engine (more natural than Piper)

> **Legacy:** the Kokoro model files (`kokoro-v1.0.int8.onnx`,
> `voices-v1.0.bin`) were removed from this setup along with the Piper
> voices. Kokoro still works if you re-download both files into
> `piper-voices/kokoro/` per the steps below. For a default setup,
> Supertonic (previous section) is faster and needs no downloads.

`dub_agent.py` can synthesize a line with either Piper or Kokoro-82M,
per character — set per speaker in `voice_map.json` the same way as any
Piper voice, just pointed at a `voices.json` alias whose entry has
`"engine": "kokoro"`. Kokoro sounds noticeably more natural than Piper
on expressive lines, at a real CPU/RAM cost, so treat it as an
opt-in upgrade for a few characters rather than a blanket replacement.

1. Install the project dependencies, including Kokoro support:
   ```
   pip install -r requirements.txt
   ```
2. Download the two model files by hand (pip can't fetch these) and
   place them in the paths configured in `voices.json`:
   - [`kokoro-v1.0.int8.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx)
     (~88MB) — the int8-quantized version, the right default on an 8GB-RAM
     laptop. If a character voiced with it sounds noticeably worse than
     Piper, try [`kokoro-v1.0.fp16.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx)
     (~169MB) instead — update the `"model"` path in `voices.json`'s
     `_kokoro` entry to match whichever you download.
   - [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)
     — the shared style-vector file every Kokoro voice reads from; place it at
     `piper-voices/kokoro/voices-v1.0.bin`.
3. Kokoro's phonemizer needs `espeak-ng` available at the system level
   (this is a real espeak-ng install, not a pip package):
   - **Windows**: install `espeak-ng-X64.msi` from
     [github.com/espeak-ng/espeak-ng/releases](https://github.com/espeak-ng/espeak-ng/releases).
     If Kokoro still can't find it, set two environment variables (System
     Properties → Environment Variables) pointing at the install folder:
     `PHONEMIZER_ESPEAK_LIBRARY` = `C:\Program Files\eSpeak NG\libespeak-ng.dll`
     and `PHONEMIZER_ESPEAK_PATH` = `C:\Program Files\eSpeak NG\espeak-ng.exe`.
   - **Mac**: `brew install espeak-ng`.
   - **Linux**: `sudo apt install espeak-ng`.
4. `voices.json` already ships a `_kokoro` entry (the shared model/voices
   paths above) and one unassigned `kokoro_test` alias (`af_bella`, a US
   female voice) so there's something to point a character at immediately.
   Run `configure_voices.py` on one episode's subtitle file and assign
   **one minor character** to `kokoro_test` — don't switch the whole
   `voice_map.json` over on the first try. Listen to that one character's
   lines before deciding whether to convert more.

   `voices.json`'s British Kokoro voices (`kokoro_bf_emma`,
   `kokoro_bf_isabella`, `kokoro_bm_george`, `kokoro_bm_fable`) are set to
   `"lang": "en-gb"` — an earlier version of this file had them on
   `"en-us"` by mistake, which mis-phonemized every line read by a GB
   voice. If you add more GB voices by hand, set `lang` to match.

   **On pacing**: Kokoro's own `speed` control is noticeably more
   sensitive than Piper's `--length-scale` — pushing it hard to force a
   line into a tight subtitle window makes that one line sound like a
   different, sped-up/slowed-down version of the character, and a run
   full of these swinging speed corrections reads as inconsistent rather
   than natural. `dub_agent.py`'s Kokoro path now only resynthesizes at a
   different pace for lines with a genuinely large timing mismatch, and
   even then only nudges speed within a narrow range — the rest of any
   gap is absorbed by ffmpeg's `atempo` (an even, pitch-preserving
   stretch) in the final duration-fit step instead. If a character still
   sounds off, it's worth checking that character isn't stuck with very
   tight subtitle timing windows across many lines — that's a translation
   pacing issue, not something a TTS setting alone will fix.

---

### Optional: Supertonic engine (44.1 kHz, fastest CPU engine)

`dub_agent.py` also supports **Supertonic** (supertone-inc's ONNX TTS, MIT
license) as a third engine, per character, same as the others. Measured on
the kind of CPU-only laptop this project targets:

- **44.1 kHz output** - the highest-fidelity of the three local engines
  (Piper: 22.05 kHz, Kokoro: 24 kHz). The dub now holds its own against the
  original audio track next to it.
- **~4x faster than realtime on CPU** - roughly 0.6s per line in practice.
  A full ~380-line episode synthesizes in about 5 minutes, where Piper takes
  15-27 minutes and Kokoro is slower still.
- The trade-off: only **ten built-in voices** (M1-M5 male, F1-F5 female),
  no per-voice model downloads, no voice cloning.

1. Install: `pip install -r requirements.txt` (adds the `supertonic`
   package). Its ~400MB model downloads automatically from HuggingFace on
   first use and is cached after that - no manual model files needed, and
   no espeak-ng requirement (unlike Kokoro).
2. `voices.json` ships aliases `supertonic_m1` ... `supertonic_f5` - assign
   a character to one with `configure_voices.py`, or by hand:
   ```json
   "MYSTERY VOICE": { "engine": "supertonic", "voice": "M3", "lang": "en",
                      "label": "Supertonic built-in male #3" }
   ```
3. Optional per-voice quality knob: `"steps": 8` (default; 5 = fastest,
   12 = highest quality). Synthesis time scales with it.

Because Supertonic outputs at 44.1 kHz natively and each line still goes
through the same two-pass pacing and tone system as the other engines,
mixing engines per character (e.g. a Supertonic lead, Kokoro secondaries,
Piper for one-line background roles) works unchanged.

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

**Quality matters more than the tone tweaks above.** A `-low` quality
voice (e.g. `southern_english_female`, the only quality that voice ships
in) will sound noticeably more robotic than a `-medium` or `-high` voice
no matter what `--noise-scale`/`--length-scale` settings it's given —
prefer `-high` for principal/frequently-speaking characters where a
`-high` version of that voice exists, and reserve `-low` voices for
minor one-line background characters. Multi-speaker models (`vctk`,
`aru`, `semaine`) are locked to speaker 0 in this repo (see Install
above) so they won't give you per-file speaker variety, just the one
voice in the file.

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

Non-interactive / scripted use is also supported — no prompts at all,
runs every pending episode found and auto-collects at the end:
```
python run.py "/path/to/Episode01.mkv" ./work
python run.py "/path/to/season_folder" ./work
```

Resume-safe at the stage level, not just per-episode: an episode whose
`.manifest.json` or `.translated.json` already exists (extraction or
script-reading finished on an earlier run) picks up from `dub_agent.py`
instead of redoing a 20-30 minute Demucs pass. This applies whether
you run `run.py` or `orchestrator.py`.

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

Per roughly 22-24 minute episode (measured on an i5-1135G7 + MX350;
with the CUDA-enabled torch install, Demucs runs on the GPU - about 9x
faster than CPU, e.g. a 5-minute clip separated in 59s vs 554s):
- Demucs extraction: about 4-6 minutes on the GPU (20-29 minutes CPU-only)
- Supertonic synthesis: about 5 minutes (Piper took 15-27; a bit more for episodes with
  several outlier-tempo lines needing the two-pass length-scale
  correction)
- **Nonverbal-gap analysis: about 13-30 minutes.** The slowest CPU step
  after Demucs, and the one people mistake for a hang. It scans the
  original vocals for laughs/gasps/sighs that no subtitle line covers,
  and each candidate clip is transcribed by Whisper ("small", CPU,
  int8) **one at a time** to check it isn't actually untranscribed
  Japanese speech. An episode with more speakers has more of these
  gaps, so the time scales with cast size, not just episode length.
  Verify it is working, not stuck, by watching the process's CPU time
  climb — it holds ~3 cores busy — and remember the log goes quiet for
  long stretches here.
- Final mux: about 1-2 minutes for plain episodes; about 4-6 minutes for
  episodes with sign lines when a hardware encoder (QSV/NVENC) is used,
  since those need a video re-encode (10-30+ minutes on software
  libx264; this step took 6 to over 12 hours before the 8-bit fix
  below — if a burn-in step is taking dramatically longer than this,
  check which encoder the run reported)
- Packing the lossless alternative (video stream-copied, see "Why the
  output is smaller than the source"): **seconds**, not minutes.
- Verify: under a minute

A full 12-episode season: roughly 9-11 hours, run back to back.

Real measured run, one 23:42 episode, demucs on GPU, 330 dialogue lines,
36 characters (total wall time roughly 50 minutes):
Demucs 3:30 / synthesis 5:14 / gap analysis ~30:00 / burn-in 4:00.

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
- `git` is not always on PATH after a Windows install either. If `git`
  is not recognised, use the full path (usually
  `"C:\Program Files\Git\cmd\git.exe"`).
- **VLC holds an open handle on a file it is playing.** Re-running a dub
  while the previous output is open in a player makes ffmpeg fail with
  `Permission denied` on the output file — at the very end of the run.
  Close the player first.

---

## Notes

- **Reading the verify report.** A finished episode's
  `.verification.json` (and `verify_agent.py`'s output) lists segments
  whose speech onset does not match the subtitle start, plus overlap
  info. On a real 358-line episode the numbers looked like: **286
  segments over the 0.3s threshold**, most only 0.3-0.6s (normal TTS
  alignment, not audible), and **24 overlapping lines** that were mixed
  together with each ducked ~3 dB. A handful of lines hit the **-0.75s
  clamp floor**, meaning they start noticeably late — those are the ones
  worth listening to if something sounds off. Presence of drift entries
  is not by itself a failure; look at the magnitude.
- **Ceiling on how expressive Piper alone can get**: Piper is a fast,
  offline, CPU-friendly TTS engine, but it has no real emotion model —
  the tone-aware delivery in `dub_agent.py` (see above) is a rule-based
  approximation layered on top of a naturally flat-sounding engine, not
  genuine voice acting. `dub_agent.py` also supports Kokoro-82M as a
  second, more natural-sounding engine, assignable per character
  alongside Piper — see "Optional: Kokoro TTS engine" above for setup.
  It's still opt-in per character rather than a blanket replacement:
  test it on one or two speaking roles before converting more of
  `voice_map.json` over to it.
- `dub_agent.py`'s remaining `atempo` correction CHAINS multiple atempo
  filters to exceed ffmpeg's single-filter range - each factor beyond 2x
  is split into another `atempo=2.0` (e.g. a 4x overrun becomes
  2.0 x 2.0), rather than silently breaking for clips over 2x their
  window. Combined with the length-scale pre-correction above, this
  should rarely be hit hard in practice now.
- A dialogue line that fails to synthesize for any reason prints a
  `left it silent` or `failed to synthesize` warning and is skipped
  rather than crashing the whole run — check the terminal output after
  a run to know which lines to spot-check.
- If a whole voice consistently fails across many lines, check its
  `model`/`config` paths in `voices.json` actually exist on disk — the
  preflight check at the start of `dub_agent.py` should now catch this
  immediately instead of letting a run finish with silent gaps.
