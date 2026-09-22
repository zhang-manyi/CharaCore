"""Unlabelled, source-checked contexts; separate from legacy DPO suites."""
import hashlib
import json
from pathlib import Path
import re

FORMAT = "characore-contexts-v1"
SPLITS = ("train", "dev", "test")
REQUIRED_FILES = {"train.json", "dev.json", "test.json", "sources.json",
                  "references.json", "exclusions.json", "evaluation_plan.json",
                  "review_index.json"}
SPEAKER = re.compile(r"^\s*([^:：]{1,30})\s*[:：]\s*(.*)$")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inside(root, path):
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Source path escapes workspace")
    return target


def policy_messages(row):
    """Allowlist only role identity and publicly visible text; never filenames/references."""
    return [
        {"role": "system", "content": (
            f"你扮演{row['character']}。根据以下可见对话与情境自然接续。"
            "不输出分析过程，不把未知事件当作已发生；按情境决定是否需要行动，不强制行动或句数。"
        )},
        {"role": "user", "content": "\n".join(
            f"{turn['speaker']}：{turn['text']}" for turn in row["visible_turns"]
        ) + "\n请接续角色的下一轮发言。"},
    ]


def validate_splits(splits, sources, formal=False):
    if set(splits) != set(SPLITS):
        raise ValueError("Explicit train/dev/test required")
    seen_ids, seen_inputs, owners = set(), set(), {}
    for split, rows in splits.items():
        if formal and not rows:
            raise ValueError("Formal split cannot be empty")
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError("Duplicate context ID")
            seen_ids.add(row["id"])
            if row.get("split") != split:
                raise ValueError("Split field mismatch")
            source = sources[row["source_id"]]
            if split != "test" and row["upstream_split"] in {"test", "ood"}:
                raise ValueError("Upstream holdout leakage")
            if row.get("exposure") == "development_review" and split == "test":
                raise ValueError("Development exposure cannot become blind test")
            if formal and row.get("review_status") != "approved":
                raise ValueError("Formal review approval missing")
            if not row["visible_turns"]:
                raise ValueError("Empty public context")
            lines = [x["source_line"] for x in row["visible_turns"]]
            if lines != sorted(set(lines)) or min(lines) < 1 or max(lines) >= row["target_line"]:
                raise ValueError("Future or unordered visible history")
            for turn in row["visible_turns"]:
                if set(turn) != {"source_line", "speaker", "text"} or not turn["text"]:
                    raise ValueError("Invalid public turn")
            for kind, key in (("family", row["family"]), ("group", row["split_group"]),
                              ("source", source["sha256"])):
                old = owners.setdefault((kind, key), split)
                if old != split:
                    raise ValueError("Scenario/source leakage across splits")
            signature = json.dumps(policy_messages(row), sort_keys=True, ensure_ascii=False)
            if signature in seen_inputs:
                raise ValueError("Duplicate policy input")
            seen_inputs.add(signature)


def build_development(root, index_path, plan_path, output):
    """Produces an integrity snapshot, never a formal training/test freeze."""
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise ValueError("Output exists; create a new version")
    index_bytes = Path(index_path).read_bytes()
    index = [json.loads(x) for x in index_bytes.decode("utf-8").splitlines() if x.strip()]
    plan = read_json(plan_path)
    splits = {s: [] for s in SPLITS}
    sources, references, exclusions = {}, {}, []
    for item in index:
        path = inside(root, item["cache_path"])
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != item["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {item['id']}")
        lines = data.decode("utf-8").splitlines()
        source_id = item["source_path"]
        sources[source_id] = dict(path=source_id, revision=item["source_revision"],
                                  sha256=actual, cache_path=item["cache_path"],
                                  kind=item["source_kind"], rights="CC BY-NC 4.0 declaration; underlying rights separate")
        if item["status"] == "quarantine":
            exclusions.append(dict(id=item["id"], source_id=source_id, reason=item["unknown"]))
            continue
        if item["status"] != "dev_review":
            raise ValueError("Unsupported review status")
        target = item["target_line"]
        if type(target) is not int or target < 1 or target > len(lines):
            raise ValueError("Invalid target line")
        target_match = SPEAKER.match(lines[target - 1])
        if not target_match or target_match[1].strip() != item["character"]:
            raise ValueError("Target speaker mismatch")
        public = []
        for number in item["visible_lines"]:
            if type(number) is not int or not 1 <= number < target:
                raise ValueError("Target/future line entered policy history")
            match = SPEAKER.match(lines[number - 1])
            if not match:
                raise ValueError("Unparsed visible turn")
            speaker, text = match[1].strip(), match[2].strip()
            override = item.get("visible_spans", {}).get(str(number))
            if override is not None:
                if not override or override not in text:
                    raise ValueError("Visible edit must be exact source substring")
                text = override
            if speaker == "旁白" and override is None:
                raise ValueError("Narration requires explicit reviewed span")
            if re.search(r"[（(].+[）)]", text) and speaker != item["character"]:
                raise ValueError("Other character parenthetical thought cannot be public")
            public.append(dict(source_line=number, speaker=speaker, text=text))
        row = {key: item[key] for key in
               ("id", "family", "split_group", "character", "target_line", "upstream_split", "exposure")}
        row.update(source_id=source_id, split="dev", review_status="development_only",
                   visible_turns=public, action_required=item.get("action_required", False))
        splits["dev"].append(row)
        # Reference includes consecutive target-speaker turns only, never given to policy.
        answer = []
        for text in lines[target - 1:]:
            match = SPEAKER.match(text)
            if not match or match[1].strip() != item["character"]:
                break
            answer.append(text)
        references[item["id"]] = dict(source_id=source_id, target_line=target,
                                     original_response=answer, known=item["known"],
                                     unknown=item["unknown"], gold=False,
                                     reference_use="audit_only_not_judge_default")
    validate_splits(splits, sources)
    output.mkdir(parents=True)
    for split, rows in splits.items():
        write_json(output / f"{split}.json", rows)
    for name, value in (("sources.json", sources), ("references.json", references),
                        ("exclusions.json", exclusions), ("evaluation_plan.json", plan),
                        ("review_index.json", index)):
        write_json(output / name, value)
    manifest = dict(format=FORMAT, status="development_snapshot",
                    formal_training_allowed=False, blind_test_available=False,
                    files={name: digest(output / name) for name in sorted(REQUIRED_FILES)},
                    input_index_sha256=hashlib.sha256(index_bytes).hexdigest())
    write_json(output / "snapshot.json", manifest)
    return dict(contexts=len(splits["dev"]), excluded=len(exclusions),
                families=len({r["family"] for r in splits["dev"]}),
                split_groups=len({r["split_group"] for r in splits["dev"]}))


def load_contexts(directory, require_formal=False):
    directory = Path(directory)
    manifest = read_json(directory / "snapshot.json")
    if manifest.get("format") != FORMAT or set(manifest.get("files", {})) != REQUIRED_FILES:
        raise ValueError("Incomplete snapshot manifest")
    if require_formal:
        raise ValueError("Development snapshot is not a formal training/evaluation freeze")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Snapshot changed: {name}")
    splits = {s: read_json(directory / f"{s}.json") for s in SPLITS}
    validate_splits(splits, read_json(directory / "sources.json"))
    return splits


def cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(build_development(root, args.index, args.plan, args.output), ensure_ascii=False))


if __name__ == "__main__":
    cli()
