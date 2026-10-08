"""Freeze a canon reference set: write freeze.json binding canon.json's raw bytes, then load it back.

The lines are transcribed from fan quote collections and still need a check against the broadcast
episodes; --sources records where they came from. The set is judge evidence only (E4).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from characore.canon import load_canon  # noqa: E402
from characore.protocol import digest, dump  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canon", type=Path, required=True, help="directory holding canon.json")
    parser.add_argument("--sources", nargs="+", required=True)
    args = parser.parse_args()
    dump(args.canon / "freeze.json", dict(canon_sha256=digest(args.canon / "canon.json"), sources=args.sources,
                                         language="Japanese original lines; scenes described in Chinese",
                                         use="judge evidence E4 only; never shown to the policy"))
    canon = load_canon(args.canon)
    print(json.dumps({k: len(v) for k, v in canon.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
