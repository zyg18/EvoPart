"""
Pack the evolvable part of the C++ project into a single text file (OpenEvolve
accepts exactly one initial-program path) and unpack it back into a build tree.

The whole partitioner is ONE evolvable file, partition.cpp - parser, data
structures, algorithm and output included. The only fixed file is the makefile
(compiler flags). The evaluator recomputes cut and balance from the written
partition file, so nothing in the program defines the objective.

Format, one block per file:

    * partition.cpp *:
    @@@
    <verbatim contents>
    @@@
"""
import os
import pathlib
import re
import shutil

# Files handed to the model.
EVOLVE_FILES = [
    "partition.cpp",
]

# Files supplied by the evaluator, never shown to the model.
FIXED_FILES = [
    "makefile",
]

_DELIM = "@@@"
_BLOCK_RE = re.compile(
    r"^\*\s*(?P<name>[^*\n]+?)\s*\*\s*:\s*\n" + _DELIM + r"\s*\n(?P<body>.*?)\n?" + _DELIM,
    re.MULTILINE | re.DOTALL,
)

PARSE_ERROR_HINT = (
    "Could not parse this file from your response. Reproduce the exact block format:\n"
    "* <path> *:\n@@@\n<full file contents>\n@@@\n"
    f"You may only write these {len(EVOLVE_FILES)} files: " + ", ".join(EVOLVE_FILES)
)


def pack(source_dir):
    """Serialise the evolvable files of `source_dir` into one string."""
    source_dir = pathlib.Path(source_dir)
    blocks = []
    for rel in EVOLVE_FILES:
        body = (source_dir / rel).read_text()
        blocks.append(f"* {rel} *:\n{_DELIM}\n{body.rstrip()}\n{_DELIM}")
    return "\n\n".join(blocks) + "\n"


def parse(text):
    """Return {relative path: contents} for every well-formed block we recognise."""
    found = {}
    for m in _BLOCK_RE.finditer(text):
        name = m.group("name").strip()
        if name in EVOLVE_FILES:
            found[name] = m.group("body")
    return found


def unpack(text, dest_dir, fixed_source_dir):
    """
    Materialise a buildable tree in `dest_dir`.

    Evolvable files come from `text`; anything the model failed to emit falls back
    to the pristine copy in `fixed_source_dir`, so a partial response still builds.
    Fixed files always come from `fixed_source_dir`.

    Returns the list of evolvable files that had to fall back.
    """
    dest_dir = pathlib.Path(dest_dir)
    fixed_source_dir = pathlib.Path(fixed_source_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse(text)
    missing = []
    for rel in EVOLVE_FILES:
        target = dest_dir / rel
        if rel in parsed:
            target.write_text(parsed[rel] + "\n")
        else:
            missing.append(rel)
            shutil.copy2(fixed_source_dir / rel, target)

    for rel in FIXED_FILES:
        shutil.copy2(fixed_source_dir / rel, dest_dir / rel)

    return missing


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).parents[1] / "initial_program")
    sys.stdout.write(pack(src))
