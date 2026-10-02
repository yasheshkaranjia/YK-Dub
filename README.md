# YK-Dub

YK-Dub creates English dubs from video files with embedded English subtitles. It extracts the audio, separates dialogue from music and effects, prepares subtitle segments, synthesizes speech using configurable character voices, mixes the dub, muxes the result with the source video, and verifies the output.

## What it does

The pipeline stages, in order:

1. Extract audio and pick the dialogue subtitle track (`extract_agent.py`)
2. Separate vocals from music/effects (Demucs)
3. Read the subtitle file into timed, per-speaker segments (`script_agent.py`)
4. Synthesize each line with the character's assigned voice (`dub_agent.py`)
5. Time-stretch, mix, and mux the dub onto the source video
6. Verify the output (`verify_agent.py`)

## Requirements

- Python 3.10 or newer
- FFmpeg and FFprobe available on `PATH`
- Enough disk space for the source video and intermediate audio files
- An OpenRouter API key when any assigned voice uses the OpenRouter engine
- A CUDA-capable GPU is optional; Demucs falls back to CPU

Demucs and PyTorch make the Python environment large. CPU separation can take roughly as long as the episode itself.

## Install

Install FFmpeg first. On Windows, for example:

```powershell
winget install ffmpeg
```

Open a new terminal and check that `ffmpeg` and `ffprobe` are available. Then create the Python environment:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, create and activate a virtual environment with `python3 -m venv venv`, then install the same `requirements.txt`.

For the OpenRouter engine, copy `.env.example` to `.env` and fill in your keys. `.env` is ignored by Git; never commit it or paste a real key into source files. The key variables are `OPENROUTER_API_KEY`, and some optional speaker-identification and reference-clip tools also use `GROQ_API_KEY`.

## Quick start

Interactive mode asks for a video or episode folder, a work directory, and whether to review character voices:

```powershell
python run.py
```

You can also pass paths directly:

```powershell
python run.py "D:\Shows\Show - 01.mkv" ".\work"
python run.py "D:\Shows\Season 1" ".\work"
```

The runner processes episodes one at a time. It skips completed episodes and continues to the next episode if one fails. `orchestrator.py` is the non-interactive batch runner:

```powershell
python orchestrator.py "D:\Shows\Season 1" ".\work"
```

`watchdog.py` can relaunch a batch after a stalled or interrupted run. See `python watchdog.py --help` for its options.

## Voices

Voice aliases and engine settings are in `voices.json`. Run the interactive voice picker on an ASS subtitle file with:

```powershell
python configure_voices.py "path\to\episode.ass"
```

The available engines include:

- **Supertonic:** local synthesis with built-in voices. It is the default engine. Its model downloads on first use; no voice WAVs are required.
- **OpenRouter preset voices:** the current `_openrouter` model in `voices.json` is Deepgram Flux. Flux uses its own preset voice IDs, such as `flux-jack-en` and `flux-meena-en`; it does not clone a voice from reference audio.
- **OpenRouter voice cloning:** the `openrouter_clone` alias uses a reference WAV. It requires an OpenRouter model that supports reference-audio cloning. Do not assign this alias while `_openrouter.model` is set to Deepgram Flux.
- **Piper and Kokoro:** legacy local engines. They require their model files to be installed separately; see the engine configuration in `voices.json` and the comments in that file.

## Voice maps (per show)

`dub_agent.py` reads `voice_map.json`, which maps each subtitle speaker name to a voice alias from `voices.json`. Speakers not in the map use `_default`, so give every character an explicit entry or a whole episode can come out in one voice.

Voice maps are local and per show. They are git-ignored so character names from one show never mix with another. Each show's map lives in `voice_maps/<show-slug>.json`, managed with `voicemaps.py`:

```powershell
python voicemaps.py use my-show     # make a show's map the active one
python voicemaps.py save my-show    # save the active map for a show
python voicemaps.py reset           # back to the empty template
python voicemaps.py check           # validate the active map
python voicemaps.py list            # list saved maps
```

On a fresh clone there is no `voice_map.json`; run `python voicemaps.py reset` (or copy `voice_map.example.json`) to create it. Without it, dubbing falls back to a single default voice.

## Output and resume

Each episode gets a folder under the selected work directory. Intermediate files include the manifest, translated segments, separated audio, and synthesis work files. The finished video is named `<episode>.dubbed.mp4`; verification results are written alongside it. Finished videos are collected into `collected/` by the runner.

With no sign text to burn in, the mux can stream-copy the video. Burning signs into the image requires video re-encoding; set `YKDUB_VIDEO_CRF` to tune quality (lower values preserve more detail and create larger files).

Existing intermediate stages are reused where possible. Set `YKDUB_RESUME_SYNTH=1` to reuse synthesized line clips after an interrupted run. Do not reuse that cache after changing voices or engines; remove that episode's `tts_work` directory first so clips from the old configuration are not mixed into the new dub.

## Useful tools

- `collect_dubbed.py` gathers completed episode videos from a work directory.
- `dub_status.py` reports pipeline progress.
- `check_cast.py` checks which speakers in an episode lack a voice assignment.
- `check_translation.py` reviews translated segment text and speakers.
- `speaker_report.py` counts lines per speaker before assigning API voices.
- `speaker_id_groq.py` can identify speakers when subtitle actor names are missing; it requires `GROQ_API_KEY`. `apply_speakers.py` writes its labels back into the episode data.
- `trim_translated.py` creates a shorter test segment from translated episode data.
- `redub.py` re-runs synthesis for finished episodes (e.g. after changing voices).
- `replace_dub_audio.py` swaps the dub track in a finished video.
- `voicemaps.py` manages per-show voice maps (see above).


## Troubleshooting

- **FFmpeg not found:** install FFmpeg and confirm both `ffmpeg` and `ffprobe` resolve from a new terminal.
- **OpenRouter key missing:** check the root `.env` spelling and ensure it contains `OPENROUTER_API_KEY=...`.
- **Clone alias rejected:** switch to a cloning-capable model or assign a preset voice. Deepgram Flux does not accept reference clips.
- **No dialogue or wrong language:** check that the source has an English dialogue subtitle track. Subtitle selection prefers English and filters signs-only tracks; without a usable subtitle track, the script stage falls back to Whisper translation.
- **Wrong sign placement:** generated `signs.ass` must retain the source `PlayResX` and `PlayResY` values.
- **Mux permission error:** close any media player holding the destination file open before rerunning.

## Project layout

- `run.py`: interactive entry point
- `orchestrator.py`: non-interactive episode and folder runner
- `watchdog.py`: batch relauncher for stalled runs
- `extract_agent.py`: audio extraction, Demucs separation, subtitle selection
- `script_agent.py`: dialogue/sign splitting and translation fallback
- `dub_agent.py`: voice synthesis, timing, mixing, and muxing
- `verify_agent.py`: post-dub timing and duration checks
- `configure_voices.py`: interactive voice picker
- `voicemaps.py`: per-show voice map manager
- `voice_map.json`: the ACTIVE speaker-to-voice map (git-ignored)
- `voice_maps/`: one saved map per show (git-ignored)
- `voices.json`: voice aliases, engines, and provider configuration
