"""Shared file integrity helpers. Labels never enter policy prompts."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    """Exclusive creation: never replace a prior request/result/failure.

    newline="" keeps bytes identical across platforms. freeze.json binds data
    files by SHA-256 of their raw bytes, so a suite frozen where dump() emits
    CRLF would fail its own integrity check wherever it is read back as LF.
    """
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
