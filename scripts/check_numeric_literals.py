"""Check C arithmetic literals and optionally wrap their numeric role."""

import argparse
from pathlib import Path
import re


_TOKEN = re.compile(
    r'(?P<skip>//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'|\b[A-Za-z_]\w*)"
    r"|(?P<number>\b0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|"
    r"\.[0-9a-fA-F]+)[pP][+-]?\d+[fFlL]?|"
    r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlL]?|"
    r"\b\d+[eE][+-]?\d+[fFlL]?)"
)


def normalize(source):
    changes = []
    for match in _TOKEN.finditer(source):
        number = match.group("number")
        if number is None or re.search(
            r"LC_(?:REAL|TIME|WIDE)_C\(\s*$", source[:match.start()]
        ):
            continue
        role = "WIDE" if number[-1:] in ("l", "L") else "REAL"
        literal = number[:-1] if number[-1:] in "fFlL" else number
        changes.append((match.start(), match.end(), f"LC_{role}_C({literal})"))
    for begin, end, value in reversed(changes):
        source = source[:begin] + value + source[end:]
    return source, len(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    total = 0
    for path in args.files:
        rewritten, count = normalize(path.read_text())
        if count:
            print(f"{path}: {count} unclassified arithmetic literals")
            total += count
            if args.fix:
                path.write_text(rewritten)
    return int(total > 0 and not args.fix)


if __name__ == "__main__":
    raise SystemExit(main())
