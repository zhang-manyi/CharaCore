"""Blind pairwise human-label set: a static HTML page plus a separate answer key.

Three kinds of pairs, each a policy reply against another reply for the same row:
  test     every held-out row where the evaluated adapter's greedy reply differs from the base reply
  train    sampled policy replies against the base reply, stratified by the judge's averaged P+R+N
           margin so ties, near ties and clear wins/losses are all present
  opener   two sampled replies from one training group, one starting with the character's fixed
           opener (嗯 / 哼) and one without, to test whether the opener itself is preferred

Sides and item order are shuffled with a fixed seed. The page shows only the card, situation,
line and the two replies; which side is which lives in key.json, which the labeller should not open.
Labels stay in the browser (localStorage) until exported as JSON.
"""
import argparse
import glob
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from characore.persona import CHARACTERS  # noqa: E402
from characore.protocol import dump  # noqa: E402
from characore.style_data import load_base_replies, load_suite  # noqa: E402

OPENER = {"rei": "嗯", "asuka": "哼"}
DIMS = ("P", "R", "N")
PAGE = Path(__file__).with_name("label_page.html")


def margin(record, policy_side):
    """Policy P+R+N minus the other candidate's, from one call; None when any score is missing."""
    if record is None or record["final"]["call_status"] != "ok":
        return None
    scores = record["final"]["judgement"]["scores"]
    other = "B" if policy_side == "A" else "A"
    if any(scores[x][d]["status"] != "scored" for x in "AB" for d in DIMS):
        return None
    return sum(scores[policy_side][d]["score"] - scores[other][d]["score"] for d in DIMS)


def training_samples(reward_dir):
    """(group key, sample, averaged AB/BA margin) for every usable judged sample of a run."""
    calls = {}
    for path in glob.glob(str(Path(reward_dir) / "rank*" / "calls" / "batch_*" / "*" / "result.json")):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        calls[record["id"]] = record
    out = []
    for path in sorted(glob.glob(str(Path(reward_dir) / "rank*" / "rewards_*.json"))):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        rejected = set(record["rejected_groups"])
        offset = record.get("row_offset", 0)
        for i, s in enumerate(record["samples"]):
            group = (Path(path).name, (offset + i) // 8)
            # v3 records carry "pairwise"; v4 records carry the same verdict as "verdict".
            s["pairwise"] = s.get("pairwise", s.get("verdict"))
            if group[1] in rejected or s["hard"] or "keys" not in s or s["pairwise"] is None:
                continue
            ab, ba = margin(calls.get(s["keys"]["AB"][:16]), "A"), margin(calls.get(s["keys"]["BA"][:16]), "B")
            if ab is not None and ba is not None:
                out.append((group, s, (ab + ba) / 2))
    return out


def pick_train(samples, count, rng):
    """Equal draws per character from five margin bands; distinct replies only."""
    bands = [(-99, -3), (-3, -0.5), (-0.5, 0.5), (0.5, 3), (3, 99)]
    chosen, seen = [], set()
    for character in ("rei", "asuka"):
        mine = [x for x in samples if x[1]["row_id"].endswith(character)]
        for k, (lo, hi) in enumerate(bands):
            pool = [x for x in mine if lo <= x[2] < hi and (x[1]["row_id"], x[1]["reply"]) not in seen]
            rng.shuffle(pool)
            want = count // 10 + (k < (count % 10) // 2)
            for _, s, m in pool[:want]:
                seen.add((s["row_id"], s["reply"]))
                chosen.append((s, m))
    return chosen


def pick_openers(samples, count, rng):
    """Pairs from one group: opener vs no opener, same row, so only the reply differs."""
    groups = {}
    for group, s, _ in samples:
        groups.setdefault(group, []).append(s)
    pairs = []
    for members in groups.values():
        character = members[0]["row_id"].rsplit("-", 1)[1]
        with_ = [s for s in members if s["reply"].startswith(OPENER[character])]
        without = [s for s in members if not s["reply"].startswith(OPENER[character])]
        if with_ and without:
            pairs.append((character, rng.choice(with_), rng.choice(without)))
    chosen = []
    for character in ("rei", "asuka"):
        mine = [p for p in pairs if p[0] == character]
        rng.shuffle(mine)
        chosen += mine[:count // 2 + (character == "rei") * (count % 2)]
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-replies", type=Path, required=True)
    parser.add_argument("--eval-replies", type=Path, required=True, help="replies.json of an adapter's test eval")
    parser.add_argument("--reward-dir", type=Path, required=True, help="reward/ of the training run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=30)
    parser.add_argument("--openers", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, test = load_suite(args.suite)
    rows = {r["id"]: r for r in train + test}
    base = load_base_replies(args.base_replies, args.suite)
    evaluated = json.loads(args.eval_replies.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    samples = training_samples(args.reward_dir)

    pairs = []  # (kind, row_id, {source: text}, extra)
    for row in test:
        if evaluated[row["id"]] != base[row["id"]]:
            pairs.append(("test", row["id"], dict(policy=evaluated[row["id"]], base=base[row["id"]]), {}))
    for s, m in pick_train(samples, args.train, rng):
        pairs.append(("train", s["row_id"], dict(policy=s["reply"], base=base[s["row_id"]]),
                      dict(v3_pairwise=s["pairwise"], v3_margin=m)))
    for _, with_, without in pick_openers(samples, args.openers, rng):
        pairs.append(("opener", with_["row_id"], dict(opener=with_["reply"], no_opener=without["reply"]), {}))

    rng.shuffle(pairs)
    items, key = [], {}
    for n, (kind, row_id, texts, extra) in enumerate(pairs, 1):
        sources = list(texts)
        rng.shuffle(sources)
        row, person = rows[row_id], CHARACTERS[rows[row_id]["character"]]
        item_id = f"q{n:03d}"
        items.append(dict(id=item_id, name=person["name"], card=person["card"], situation=row["situation"],
                          speaker=row["speaker"], line=row["line"],
                          left=texts[sources[0]], right=texts[sources[1]]))
        key[item_id] = dict(kind=kind, row_id=row_id, left=sources[0], right=sources[1], **extra)

    args.output.mkdir(parents=True, exist_ok=False)
    set_id = f"{args.output.name}-seed{args.seed}"
    payload = json.dumps(dict(set_id=set_id, items=items), ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.read_text(encoding="utf-8").replace("/*LABEL_DATA*/null", payload)
    with (args.output / "label.html").open("x", encoding="utf-8", newline="") as stream:
        stream.write(html)
    dump(args.output / "key.json", dict(set_id=set_id, eval_replies=str(args.eval_replies),
                                        reward_dir=str(args.reward_dir), items=key))
    counts = {k: sum(p[0] == k for p in pairs) for k in ("test", "train", "opener")}
    print(json.dumps(dict(set_id=set_id, items=len(items), **counts, page=str(args.output / "label.html"))))


if __name__ == "__main__":
    main()
