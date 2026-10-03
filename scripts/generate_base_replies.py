"""Generate the frozen base replies once: greedy, untrained base model, every suite row.

The pairwise reward compares each policy sample against these, so they are
generated a single time, hashed, and bound to the suite. Training refuses a
reply set whose bytes or suite binding changed.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from characore.protocol import digest, dump
from characore.style_data import load_base_replies, load_suite
from characore.style_rewards import hard_violation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    train, test = load_suite(args.suite)
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import set_seed
    from characore.model_loader import load_model
    from characore.precision import select_precision
    from characore.style_generate import first_clean, generate

    set_seed(17)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = select_precision(device, args.precision)
    model, tokenizer = load_model(args.base, device=device, precision=precision)
    model.config.use_cache = True
    rows = train + test
    replies = first_clean(generate(model, tokenizer, rows, args.max_new_tokens, args.batch_size))
    empty = [k for k, v in replies.items() if not v]
    if empty:
        raise ValueError(f"base produced empty replies for {empty[:5]}; refusing to freeze")
    dump(args.output / "replies.json", replies)
    by_id = {r["id"]: r for r in rows}
    hard = {}
    for key, reply in replies.items():
        reason = hard_violation(reply, by_id[key]["character"])
        if reason:
            hard[reason] = hard.get(reason, 0) + 1
    dump(args.output / "freeze.json", dict(
        replies_sha256=digest(args.output / "replies.json"),
        suite_freeze_sha256=digest(args.suite / "freeze.json"),
        base=str(args.base), base_files_sha256={p.name: digest(p) for p in sorted(Path(args.base).iterdir())
                                                if p.is_file() and p.suffix in (".json", ".safetensors")},
        decoding=dict(do_sample=False, max_new_tokens=args.max_new_tokens, enable_thinking=False,
                      use_model_defaults=False),
        checkpoint_generation_config=model.generation_config.to_dict(),
        device=device, precision=precision, rows=len(replies), base_hard_violations=hard,
        packages={p: importlib.metadata.version(p) for p in ("torch", "transformers")}))
    load_base_replies(args.output, args.suite)
    print(json.dumps(dict(rows=len(replies), base_hard_violations=hard,
                          examples=dict(list(replies.items())[:4])), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
