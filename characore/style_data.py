"""Frozen suite and frozen base-reply loading. Every file is bound by SHA-256 of its raw bytes."""
from pathlib import Path

from characore.persona import CHARACTERS, persona_identity
from characore.protocol import digest, read_json

ROW_KEYS = {"id", "template", "character", "situation", "speaker", "line"}


def load_suite(path):
    """Returns (train, test). Refuses a suite whose bytes, personas or split drifted."""
    path = Path(path)
    freeze = read_json(path / "freeze.json")
    if set(freeze["files"]) != {"train.json", "test.json"}:
        raise ValueError("suite must freeze exactly train.json and test.json")
    for name, expected in freeze["files"].items():
        if digest(path / name) != expected:
            raise ValueError(f"{name} does not match freeze.json")
    if freeze["persona_sha256"] != persona_identity():
        raise ValueError("persona cards changed since the suite was frozen")
    train, test = read_json(path / "train.json"), read_json(path / "test.json")
    for row in train + test:
        if set(row) != ROW_KEYS or row["character"] not in CHARACTERS:
            raise ValueError(f"bad suite row {row.get('id')}")
    ids = [r["id"] for r in train + test]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row id")
    if {r["template"] for r in train} & {r["template"] for r in test}:
        raise ValueError("train and test share a template")
    return train, test


def select_character(rows, character):
    """One character's rows for a per-character run; None keeps both."""
    if character is None:
        return rows
    if character not in CHARACTERS:
        raise ValueError(f"unknown character {character}")
    return [r for r in rows if r["character"] == character]


def load_base_replies(path, suite):
    """Frozen base replies: every suite row exactly once, bound to this suite's freeze."""
    path = Path(path)
    freeze = read_json(path / "freeze.json")
    if digest(path / "replies.json") != freeze["replies_sha256"]:
        raise ValueError("replies.json does not match its freeze.json")
    if freeze["suite_freeze_sha256"] != digest(Path(suite) / "freeze.json"):
        raise ValueError("base replies were generated for a different suite")
    replies = read_json(path / "replies.json")
    train, test = load_suite(suite)
    if set(replies) != {r["id"] for r in train + test}:
        raise ValueError("base replies must cover every suite row exactly once")
    if any(not isinstance(v, str) for v in replies.values()):
        raise ValueError("base replies must be strings")
    return replies
