"""Cache and inspect pinned upstream text; never generate labels or model calls."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "290bf4ad22076156083804013012847a77c0646c"
BASE = "https://raw.githubusercontent.com/LC1332/Chat-Haruhi-Suzumiya/" + REVISION + "/"
ROLES = {"ayaka": "神里绫华", "zhongli": "钟离", "hutao": "胡桃"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pinned_tree():
    folder = ROOT / "data/raw/genshin_selection_20260921"
    data = (folder / "tree.json").read_bytes()
    manifest = [json.loads(x) for x in (folder / "sources.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = next(x["sha256"] for x in manifest if x["path"] == "tree.json")
    if sha(data) != expected:
        raise ValueError("Cached tree SHA-256 mismatch")
    tree = json.loads(data)
    if tree.get("truncated") or tree["sha"] != REVISION:
        raise ValueError("Wrong or incomplete upstream tree")
    return [x for x in tree["tree"] if re.fullmatch(
        r"characters/(ayaka|zhongli|hutao)/texts/[^/]+\.txt", x["path"])]


def collect(output, download=False):
    if output.exists():
        raise ValueError("Output exists; use a new output directory")
    entries = pinned_tree()
    output.mkdir(parents=True)
    cache = ROOT / "data/raw/genshin_texts_v1"
    cache.mkdir(parents=True, exist_ok=True)
    old = ROOT / "data/raw/genshin_selection_20260921"
    reuse = ROOT / "data/raw/reuse_audit_20260921"

    def one(entry):
        path = entry["path"]
        destination = cache / path
        candidates = [destination, old / path]
        if path in {"characters/ayaka/texts/24.txt", "characters/ayaka/texts/25.txt"}:
            candidates.append(reuse / ("ayaka_" + Path(path).name))
        result = dict(path=path, git_blob=entry["sha"], expected_bytes=entry["size"])
        try:
            origin = next((p for p in candidates if p.exists()), None)
            if origin is not None:
                data = origin.read_bytes()
                mode = "reused"
            elif download:
                request = urllib.request.Request(BASE + urllib.parse.quote(path),
                                                 headers={"User-Agent": "CharaCore-source-preparation"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read(1024 * 1024 + 1)
                mode = "downloaded"
            else:
                return dict(result, status="missing")
            if len(data) != entry["size"] or blob(data) != entry["sha"]:
                raise ValueError("Upstream size or Git blob mismatch")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(data)
            return dict(result, status="ok", mode=mode, bytes=len(data), sha256=sha(data),
                        cache_path=destination.relative_to(ROOT).as_posix(),
                        url=BASE + urllib.parse.quote(path))
        except Exception as error:
            return dict(result, status="error", error=str(error))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, entries))
    write_json(output / "sources.json", results)
    good = [x for x in results if x["status"] == "ok"]
    groups = defaultdict(list)
    speakers = re.compile(r"^([^:：]{1,30})\s*[:：]")
    inventory = []
    for item in good:
        groups[item["sha256"]].append(item["path"])
        raw = (ROOT / item["cache_path"]).read_text(encoding="utf-8")
        lines = raw.splitlines()
        role_id = item["path"].split("/")[1]
        name = Path(item["path"]).name
        turns = []
        for i, line in enumerate(lines, 1):
            match = speakers.match(line.strip())
            if match:
                turns.append({"line": i, "speaker": match[1].strip(), "text": line})
        profile = any(x in name.lower() for x in ("prompt", "设定"))
        voice = bool(re.match(r"^(关于|闲聊|初次见面|想要了解|喜欢|讨厌|下雨|下雪|打雷|刮大风|雨过|阳光|起风|早上|中午|晚上|晚安|生日|收到|突破|元素|倒下|生命值|同伴|加入|冲刺|打开|重受击|有什么|感兴趣)", name))
        kind = "profile_or_setting" if profile else "voice_candidate" if voice else "dialogue_candidate"
        review_flags = []
        if not any(x["speaker"] == ROLES[role_id] for x in turns):
            review_flags.append("no_target_speech")
        if any("旁白" == x["speaker"] for x in turns):
            review_flags.append("narration_needs_visibility_review")
        if any(re.search(r"[（(].+[）)]", x["text"]) for x in turns):
            review_flags.append("parenthetical_visibility_review")
        inventory.append(dict(item, role=role_id, kind=kind, line_count=len(lines),
                              speakers=sorted({x["speaker"] for x in turns}),
                              flags=review_flags, status="unreviewed",
                              # Entire character is the conservative unresolved group.
                              split_group="unresolved_" + role_id, split="quarantine"))
    write_json(output / "inventory.json", inventory)
    write_json(output / "duplicates.json", [v for v in groups.values() if len(v) > 1])
    # Exact shared substantial lines suggest merging; they do not prove independence.
    shared = defaultdict(set)
    for item in inventory:
        text = (ROOT / item["cache_path"]).read_text(encoding="utf-8")
        for line in text.splitlines():
            normalized = re.sub(r"[\s「」『』：:，。！？、…·\-]", "", line)
            if len(normalized) >= 24:
                shared[normalized].add(item["path"])
    write_json(output / "overlap_candidates.json",
               [sorted(paths) for paths in shared.values() if len(paths) > 1])
    summary = dict(revision=REVISION, expected_files=len(entries), verified_files=len(good),
                   unique_contents=len(groups), total_bytes=sum(x["bytes"] for x in good),
                   downloaded_files=sum(x.get("mode") == "downloaded" for x in good),
                   reused_files=sum(x.get("mode") == "reused" for x in good),
                   errors=[x for x in results if x["status"] != "ok"],
                   roles=dict(Counter(x["role"] for x in inventory)),
                   kind_counts=dict(Counter(x["kind"] for x in inventory)),
                   scene_count=None, independent_scene_count_verified=False,
                   grouping_status="All unreviewed entries quarantined; overlap hints only")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return bool(summary["errors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    sys.exit(collect(args.output, args.download_missing))


if __name__ == "__main__":
    main()
