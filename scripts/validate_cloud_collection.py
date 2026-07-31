from __future__ import annotations

import argparse
import json
from pathlib import Path

from yellowstone.validate_cloud_collection import validate_workers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dirs", nargs="+")
    parser.add_argument("--expected-games", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_workers([Path(p) for p in args.replay_dirs], args.expected_games)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
