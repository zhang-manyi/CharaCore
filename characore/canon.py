"""Reference lines from the original work, shown to the judge only (evidence E4).

The judge no longer has to rely on its own memory of the characters: a short, fixed set of
canon lines with their scenes is part of every pairwise request. The policy never sees them,
so its prompt and the frozen base replies are unchanged. The set is bound by SHA-256 like the
suites; freeze.json records where each line was transcribed from.
"""
from pathlib import Path

from characore.persona import CHARACTERS
from characore.protocol import digest, read_json

ENTRY_KEYS = {"episode", "context", "line"}


def load_canon(path):
    """{character: [entry, ...]} after checking the bytes against freeze.json and the schema."""
    path = Path(path)
    freeze = read_json(path / "freeze.json")
    if digest(path / "canon.json") != freeze["canon_sha256"]:
        raise ValueError("canon.json does not match its freeze.json")
    canon = read_json(path / "canon.json")
    if set(canon) != set(CHARACTERS):
        raise ValueError("canon must list exactly the suite's characters")
    for character, entries in canon.items():
        if not entries or any(set(e) != ENTRY_KEYS or not all(isinstance(e[k], str) and e[k].strip() for k in ENTRY_KEYS)
                              for e in entries):
            raise ValueError(f"bad canon entries for {character}")
    return canon


def canon_identity(path):
    """What a run binds: the hash of the frozen file, or None for a run without reference lines."""
    return None if path is None else read_json(Path(path) / "freeze.json")["canon_sha256"]


def render(entries):
    """One line per entry: episode, scene, then the original line."""
    return "\n".join(f"{e['episode']}｜{e['context']}｜「{e['line']}」" for e in entries)
