"""
Patches fairseq's dataclass/configs.py so it works on Python 3.11+.

Why this is needed: Python 3.11 tightened dataclasses to reject a
mutable dataclass instance as a field's *default value* (e.g.
`common: CommonConfig = CommonConfig()`), requiring
`field(default_factory=CommonConfig)` instead. fairseq 0.12.2 (the
version rvc-python pins) predates this and was never updated, so
`import fairseq` raises:
    ValueError: mutable default <class '...'> for field ... is not
    allowed: use default_factory

This script finds every `name: Type = Type()`-shaped field in
configs.py and rewrites it to use `field(default_factory=Type)`,
then makes sure `field` is imported from dataclasses. It keeps a
`.bak` of the original file and is safe to re-run (skips already-
patched lines).

Usage (run with your venv activated, from anywhere):
    python patch_fairseq_py311.py
"""
import re
import subprocess
import sys
from pathlib import Path


def find_configs_py() -> Path:
    result = subprocess.run(
        [sys.executable, "-c", "import fairseq, os; print(os.path.dirname(fairseq.__file__))"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # fairseq itself can't even be located this way if the dataclass
        # import already fails at package init - fall back to a direct guess
        # relative to the venv's site-packages.
        import site
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            candidate = Path(sp) / "fairseq" / "dataclass" / "configs.py"
            if candidate.exists():
                return candidate
        print("Could not locate fairseq's configs.py automatically.")
        print("Find it manually (likely venv\\Lib\\site-packages\\fairseq\\dataclass\\configs.py)")
        print("and re-run with its path as an argument.")
        sys.exit(1)
    fairseq_dir = Path(result.stdout.strip())
    return fairseq_dir / "dataclass" / "configs.py"


# Matches lines like:  "    common: CommonConfig = CommonConfig()"
# Captures: indent, field name+type annotation, class name being
# default-constructed. Only matches zero-arg constructor calls that
# exactly mirror the annotation's class name pattern (safe/conservative -
# won't touch unrelated `= SomeFunc()` calls with args).
FIELD_PATTERN = re.compile(
    r'^(?P<indent>\s+)(?P<name>\w+):\s*(?P<type>[\w\.]+)\s*=\s*(?P<ctor>[\w\.]+)\(\)\s*$'
)


def patch_file(path: Path) -> int:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    changed = 0
    out_lines = []
    for line in lines:
        m = FIELD_PATTERN.match(line)
        if m and m.group("ctor")[0].isupper():
            # Only patch when it looks like a dataclass default (capitalized
            # constructor name), not e.g. `x: int = field()` or `y: str = ""`.
            new_line = f'{m.group("indent")}{m.group("name")}: {m.group("type")} = field(default_factory={m.group("ctor")})\n'
            out_lines.append(new_line)
            changed += 1
        else:
            out_lines.append(line)

    if changed == 0:
        return 0

    patched = "".join(out_lines)

    # Make sure `field` is imported from dataclasses
    if "from dataclasses import" in patched and "field" not in patched.split("\n")[0:20].__str__():
        patched = re.sub(
            r'from dataclasses import ([^\n]+)',
            lambda m: f'from dataclasses import {m.group(1)}' + ("" if "field" in m.group(1) else ", field"),
            patched,
            count=1,
        )
    elif "from dataclasses import" not in patched:
        patched = "from dataclasses import field\n" + patched

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"Backed up original to: {backup}")
    else:
        print(f"Backup already exists at {backup} (not overwriting)")

    path.write_text(patched, encoding="utf-8")
    return changed


def main():
    if len(sys.argv) > 1:
        configs_py = Path(sys.argv[1])
    else:
        configs_py = find_configs_py()

    if not configs_py.exists():
        print(f"File not found: {configs_py}")
        sys.exit(1)

    print(f"Patching: {configs_py}")
    n = patch_file(configs_py)
    if n == 0:
        print("No unpatched mutable-default fields found (already patched, or pattern didn't match).")
    else:
        print(f"Patched {n} field(s).")
        print("\nNow re-test with:")
        print('    python -c "import rvc_python; print(\'rvc-python import OK\')"')


if __name__ == "__main__":
    main()
