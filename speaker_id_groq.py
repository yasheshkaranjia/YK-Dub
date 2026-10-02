"""Identify the speaker of every dialogue line with Groq, using the
surrounding lines as context.

    python speaker_id_groq.py "<path to .translated.json>" [--model NAME]

WHY THIS IS NEEDED
The release this runs against has NO speaker information at all: every one
of its 400 dialogue lines has a blank ASS "Name" field. YK-Dub's normal
casting path (configure_voices.py) reads that field, so with it empty the
whole episode collapses to a single unnamed speaker - there is nothing to
assign voices to, and no way to put one character on a different engine.

WHY AN LLM CAN DO THIS AND PUNCTUATION RULES CANNOT
Who is speaking is not in the text of the line itself - it is in the
conversation around it. "What happened now?" is meaningless alone, but
after "Please return to the castle." / "I told you not to sneak up on me,
Bob." it is clearly a back-and-forth and the two lines have different
speakers. That is context reasoning, which is what an LLM is for.

WHY THE LINES ARE SENT IN CONTEXT WINDOWS, NOT ONE AT A TIME
A single line alone is genuinely ambiguous, so the script sends a window of
neighbouring lines and asks for the speaker of each - the model sees the
conversation, not isolated sentences. It also keeps the request count low,
which matters because the layer above it is rate-limited.

OUTPUT
Writes <stem>.speakers.json next to the input: a list of
{"index", "start", "text", "speaker"} plus a summary count per speaker.
Nothing is modified - the input JSON is only read.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"   # strongest general model Groq serves
WINDOW = 24                              # neighbouring lines per request

SYSTEM_PROMPT = """You are labelling who speaks each line of dialogue in an
English-subtitled anime episode. You receive a numbered list of consecutive
lines. For EACH line, output the speaking character's name.

Rules:
- Respond with ONLY one label per line, in this exact format:
  12=Bob
  13=Prince Carl
  14=Bob
  (one per line, no other text, no commentary, no quoting the dialogue)
- Use a consistent single spelling for each character across the whole
  episode. Reuse exactly the name you used before for the same speaker.
- Multiple lines by the same character in a row must all carry the same name.
- Use a short name (1-3 words), e.g. "Bob", "Carol", "Dana", "Narrator".
- A line addressed TO someone is not spoken BY them: 'Sir Carol...' is
  spoken by whoever is talking, not by Carol. Use the conversation to tell.
- If a line is narration with no character on screen, label it "Narrator".
- If a line is genuinely a crowd or several people at once, label it "Crowd".
- Never invent a name that is not supported by the conversation."""


def load_env(path: Path = Path(".env")) -> None:
    """Real environment wins, so an exported key is never overridden."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def parse_labels(text: str, expected_indices: set) -> dict:
    """Pulls 'N=Name' pairs out of a chatty response.

    Deliberately forgiving: models wrap answers in prose, markdown fences or
    trailing explanation even when told not to, and discarding a whole
    window over that would lose 24 lines for a cosmetic reason. Anything
    that parses as '<digits>=<name>' is taken; anything missing is left
    unlabelled for the next window to cover rather than guessed at."""
    found = {}
    for match in re.finditer(r"^\s*(\d+)\s*=\s*(.+?)\s*$", text, re.MULTILINE):
        index, name = int(match.group(1)), match.group(2).strip()
        # Some models add a parenthetical: 'Bob (the butler)'
        name = re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", name).strip(" \"'.:-")
        if index in expected_indices and name and len(name) <= 40:
            found[index] = name
    return found


def identify(segments: list, api_key: str, model: str, window_size: int) -> dict:
    """Returns {segment index -> speaker name}.

    window_size is passed in rather than read from a module-level global:
    it is genuinely per-call configuration, and a global here would need a
    `global` statement that shadows the value it is trying to set."""
    labels = {}
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    total_windows = (len(segments) + window_size - 1) // window_size
    for window_no, start in enumerate(range(0, len(segments), window_size), 1):
        window = segments[start:start + window_size]
        indices = list(range(start, start + len(window)))
        listing = "\n".join(f"{start + offset}={seg['final_text'].strip()}"
                            for offset, seg in enumerate(window))
        # Carrying the names already chosen stops the same character being
        # spelled differently in later windows ('Bob' vs 'Bobby'),
        # which would otherwise shatter one character into several voices.
        known = sorted(set(labels.values()))
        user = listing
        if known:
            user += ("\n\nNames already used earlier in this episode "
                     "(reuse these exact spellings): " + ", ".join(known))

        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    GROQ_URL, headers=headers, timeout=180,
                    json={"model": model, "temperature": 0,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": user}]},
                )
                if resp.status_code == 429:
                    wait = 20 * attempt
                    print(f"  [groq] rate limited on window {window_no}, "
                          f"waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"  [groq] HTTP {resp.status_code}: {resp.text[:200]}",
                          flush=True)
                    break
                content = resp.json()["choices"][0]["message"]["content"]
                got = parse_labels(content, set(indices))
                labels.update(got)
                print(f"  [groq] window {window_no}/{total_windows} "
                      f"(lines {start}-{indices[-1]}): {len(got)}/{len(window)} labelled",
                      flush=True)
                break
            except Exception as e:
                print(f"  [groq] {type(e).__name__} on window {window_no} "
                      f"attempt {attempt}: {e}", flush=True)
                time.sleep(5 * attempt)

    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("translated_json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="neighbouring lines sent per request")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set (checked .env and environment)")

    src = Path(args.translated_json)
    data = json.loads(src.read_text(encoding="utf-8"))
    segments = data["segments"]
    print(f"[groq] {src.name}: {len(segments)} lines, model={args.model}, "
          f"window={args.window} (~{(len(segments) + args.window - 1) // args.window} requests)")

    labels = identify(segments, api_key, args.model, args.window)

    out = []
    for i, seg in enumerate(segments):
        out.append({
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["final_text"].strip(),
            "speaker": labels.get(i, ""),
        })

    counts = Counter(r["speaker"] or "(unlabelled)" for r in out)
    result = {
        "_source": str(src),
        "_model": args.model,
        "_labelled": sum(1 for r in out if r["speaker"]),
        "_total": len(out),
        "speakers": counts.most_common(),
        "lines": out,
    }
    dest = src.with_name(src.name.replace(".translated.json", ".speakers.json"))
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[groq] labelled {result['_labelled']}/{result['_total']} lines")
    print(f"[groq] wrote {dest.name}")
    print(f"\n{'lines':>6}  {'share':>7}  speaker")
    print("-" * 50)
    for name, count in counts.most_common():
        print(f"{count:6d}  {100.0 * count / len(out):6.1f}%  {name}")


if __name__ == "__main__":
    main()