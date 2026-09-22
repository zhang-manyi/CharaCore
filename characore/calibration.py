"""Prepare offline, blinded human calibration materials. Never calls a model."""
import copy
import json
from pathlib import Path
import random

from characore.contexts import digest, load_contexts, read_json, write_json
from characore.judge import DIMS, PROTOCOL, SYSTEM, export_requests, make_request


def build_packet(suite, cases_path, output, seed=20260922):
    suite, output = Path(suite), Path(output)
    if output.exists():
        raise ValueError("Output exists; use a new run directory")
    splits = load_contexts(suite)
    rows = {row["id"]: row for row in splits["dev"]}
    plan = read_json(suite / "evaluation_plan.json")
    if plan["judge_protocol"] != PROTOCOL:
        raise ValueError("Human packet requires v0.3 neutral evidence IDs")
    cases = read_json(cases_path)
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Nonempty unique calibration cases required")
    if len(cases) != plan["calibration"]["planned_unique_pairs"]:
        raise ValueError("Case count differs from calibration plan")
    seen = set()
    for case in cases:
        if case["context_id"] not in rows:
            raise ValueError("Calibration must use dev contexts only")
        if case["expected"] not in {"A", "B", "tie", "insufficient"}:
            raise ValueError("Invalid design expectation")
        pair = (case["context_id"], tuple(sorted(case["candidates"].values())))
        if pair in seen:
            raise ValueError("Duplicate or reversed duplicate comparison")
        seen.add(pair)
        make_request(rows[case["context_id"]], case["candidates"], plan["rubric"])

    rng = random.Random(seed)
    order = list(range(len(cases)))
    rng.shuffle(order)
    reversals = [i % 2 == 1 for i in range(len(cases))]
    rng.shuffle(reversals)
    blind, audit, annotations = [], {}, []
    for number, (index, reverse) in enumerate(zip(order, reversals), 1):
        case = cases[index]
        row = rows[case["context_id"]]
        neutral_id = f"review_{number:03d}"
        request = make_request(row, case["candidates"], plan["rubric"], reverse)
        payload = json.loads(request["messages"][1]["content"])
        blind.append(dict(id=neutral_id, material=payload))
        audit[neutral_id] = dict(
            case=case, context_id=row["id"], family=row["family"],
            split_group=row["split_group"], source_id=row["source_id"],
            display_to_original=request["display_to_original"],
            evidence_lines={f"E{i}": turn["source_line"]
                            for i, turn in enumerate(row["visible_turns"], 1)},
        )
        annotations.append(dict(
            id=neutral_id, winner=None, reason="", preference_evidence_ids=[],
            scores={side: {dim: dict(status=None, score=None, reason="", evidence_ids=[])
                           for dim in DIMS} for side in ("A", "B")},
            unsupported_citations=[], notes="", review_status="pending",
        ))
    sources = read_json(suite / "sources.json")
    used_rows = [rows[key] for key in sorted({case["context_id"] for case in cases})]
    evidence_index = [
        dict(context_id=row["id"], source=sources[row["source_id"]],
             target_line=row["target_line"], visible_turns=row["visible_turns"],
             split_group=row["split_group"], exposure=row["exposure"],
             claim_scope="Only visible statements and local stance; no inferred stable preferences",
             stable_preferences_verified=[], review_origin="assistant_not_human")
        for row in used_rows
    ]
    # Everything is validated before creating the new output directory.
    export_requests(suite, cases_path, output / "judge")
    (output / "human").mkdir()
    (output / "audit").mkdir()
    instructions = (
        "仅根据材料独立标注。先看证据再评候选，不使用角色外部知识；"
        "不读取audit或模型结果。status为scored/insufficient/not_applicable，"
        "scored用0至4整数，其他用null；not_applicable只允许非必需行动的A维。"
        "winner为A/B/tie/insufficient，严格偏好引用证据；不确定可保留。"
        "完成后填写reviewer、completed_at并将review_status设为completed。"
    )
    write_json(output / "human" / "review.json",
               dict(protocol=PROTOCOL, instructions=instructions, judge_instructions=SYSTEM, cases=blind))
    for reviewer in ("reviewer_1", "reviewer_2"):
        write_json(output / "human" / f"{reviewer}.json",
                   dict(protocol=PROTOCOL, reviewer=None, completed_at=None,
                        annotations=copy.deepcopy(annotations)))
    write_json(output / "audit" / "mapping_and_expectations.json", audit)
    write_json(output / "audit" / "visible_evidence_index.json", evidence_index)
    schedule = [case["id"] + suffix for case in cases for suffix in ("_AB", "_BA")]
    rng.shuffle(schedule)
    write_json(output / "audit" / "request_schedule.json",
               dict(seed=seed, request_ids=schedule,
                    note="Future calls must be independent; only messages are sent"))
    counts = {label: sum(case["expected"] == label for case in cases)
              for label in ("A", "B", "tie", "insufficient")}
    manifest = dict(
        status="offline_unreviewed_calibration_design", protocol=PROTOCOL,
        suite_sha256=digest(suite / "snapshot.json"), cases_sha256=digest(cases_path),
        seed=seed, comparison_count=len(cases), context_count=len(used_rows),
        family_count=len({row["family"] for row in used_rows}),
        conservative_split_group_count=len({row["split_group"] for row in used_rows}),
        independent_scene_count_verified=False, design_expectation_counts=counts,
        calls_executed=0, human_reviews_completed=0, calibration_passed=False,
        formal_freeze_created=False,
        files={path.relative_to(output).as_posix(): digest(path)
               for path in sorted(output.rglob("*.json"))},
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_packet(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    actual_files = {p.relative_to(directory).as_posix()
                    for p in directory.rglob("*.json") if p != directory / "manifest.json"}
    if actual_files != set(manifest["files"]):
        raise ValueError("Packet file set changed")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError("Packet changed: " + name)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_packet(args.suite, args.cases, args.output)
    print(json.dumps({key: result[key] for key in (
        "comparison_count", "context_count", "conservative_split_group_count",
        "calls_executed", "human_reviews_completed")}))
