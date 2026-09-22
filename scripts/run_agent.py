"""Run/replay one offline development task. Scripted mode requires only Python."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.agent import ScriptedPolicy, dump, replay, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    run = subs.add_parser("run")
    run.add_argument("--policy", choices=("scripted", "local"), default="scripted")
    run.add_argument("--scenario", choices=("success", "illegal", "commitment"), default="success")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-steps", type=int, default=12)
    run.add_argument("--base", type=Path)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--quantize", action="store_true")
    run.add_argument("--max-new-tokens", type=int, default=192)
    run.add_argument("--seed", type=int, default=17)
    check = subs.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    playback = subs.add_parser("replay")
    playback.add_argument("trajectory", type=Path)
    args = parser.parse_args()
    if args.mode == "replay":
        trace = replay(args.trajectory)
        for row in trace["records"]:
            print(json.dumps(dict(turn=row["turn"], output=row["parsed_output"],
                                  result=row["tool_result"], changes=row["state_changes"],
                                  failure=row["failure"]), ensure_ascii=False))
        print(json.dumps(dict(replay_verified=True, **trace["metrics"]), ensure_ascii=False))
        return 0
    # Reserve before loading: initialization failures also get their own permanent record.
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "invocation.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        if args.mode == "verify":
            results = {}
            for scenario in ("success", "illegal", "commitment"):
                trace = run_episode(ScriptedPolicy(scenario), args.output / scenario)
                replay(args.output / scenario / "trajectory.json")
                m = trace["metrics"]
                if not m["task_completed"] or m["rejected_action_state_mutations"] != 0:
                    raise AssertionError(f"mechanism verification failed: {scenario}")
                expected_rejections = {"success": 0, "illegal": 2, "commitment": 1}[scenario]
                if m["tool_rejections"] != expected_rejections or not m["commitment_kept"]:
                    raise AssertionError(f"unexpected rejection/commitment: {scenario}")
                results[scenario] = m
            dump(args.output / "verification.json", dict(claim="软件机制验证，不是基座表现。",
                                                         passed=True, scenarios=results))
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0
        if args.policy == "local":
            if args.base is None:
                raise ValueError("local policy requires --base")
            if args.scenario != "success":
                raise ValueError("--scenario selects scripted sequences only")
            from characore.local_policy import LocalPolicy
            policy = LocalPolicy(args.base, args.device, args.quantize, args.max_new_tokens, args.seed)
        else:
            policy = ScriptedPolicy(args.scenario)
        trace = run_episode(policy, args.output / "episode", args.max_steps)
        replay(args.output / "episode" / "trajectory.json")
        print(json.dumps(trace["metrics"], ensure_ascii=False, indent=2))
        return 0 if trace["metrics"]["task_completed"] else 2
    except Exception as exc:
        dump(args.output / "failure.json", dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        print(f"Failed; preserved details in {args.output / 'failure.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
