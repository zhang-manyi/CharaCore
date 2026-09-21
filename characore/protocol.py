"""Frozen scenario contract. Labels and evidence never enter policy prompts."""
import hashlib
import json
from pathlib import Path



def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_suite(directory):
    """Load an explicitly selected, versioned data suite; no bundled dataset."""
    directory = Path(directory)
    freeze = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in freeze["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Frozen evaluation changed: {name}; create a new suite version")
    manifest = json.loads((directory / "train_manifest.json").read_text(encoding="utf-8"))
    if digest(directory / "freeze.json") != manifest["evaluation_freeze_sha256"]:
        raise ValueError("Training data was built for a different evaluation freeze")
    if digest(directory / "train.json") != manifest["train_sha256"]:
        raise ValueError("Training manifest mismatch")
    profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
    train = json.loads((directory / "train.json").read_text(encoding="utf-8"))
    test = json.loads((directory / "eval.json").read_text(encoding="utf-8"))
    validate(train, test)
    return profiles, train, test


def validate(train, test):
    if {r["family"] for r in train} & {r["family"] for r in test}:
        raise ValueError("Scenario-family leakage")
    ids, inputs = set(), set()
    for r in train + test:
        if r["id"] in ids:
            raise ValueError("Duplicate ID")
        ids.add(r["id"])
        signature = json.dumps({k: r[k] for k in ["character", "goal", "relationship", "history", "situation", "user"]}, sort_keys=True)
        if signature in inputs:
            raise ValueError("Duplicate scenario input")
        inputs.add(signature)
        if r["preference"] not in {"a", "b", "tie", "insufficient"} or not r["evidence"]:
            raise ValueError("Invalid or unsupported preference")
        if r["candidates"]["a"] == r["candidates"]["b"] and r["preference"] in {"a", "b"}:
            raise ValueError("Identical candidates cannot have a strict preference")


def prompt_messages(row, profiles):
    p = profiles[row["character"]]
    system = (f"你是{p['name']}，{p['role']}。稳定偏好：{p['stable_preferences']}表达风格：{p['style']}"
              "临时情绪不会自动改变稳定偏好。请结合目标、关系、相关历史和当前证据决定行动。"
              "直接用一到三句台词说明你现在采取的具体行动，不输出分析过程。")
    context = {k: row[k] for k in ["goal", "relationship", "history", "emotion", "situation", "user"]}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]


def render_prompt(row, profiles, tokenizer):
    return tokenizer.apply_chat_template(prompt_messages(row, profiles), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def dpo_records(rows, profiles, tokenizer):
    result = []
    for row in rows:
        if row["preference"] not in {"a", "b"}:
            continue
        chosen = row["preference"]
        rejected = "b" if chosen == "a" else "a"
        result.append({"prompt": render_prompt(row, profiles, tokenizer),
                       "chosen": row["candidates"][chosen], "rejected": row["candidates"][rejected]})
    return result
