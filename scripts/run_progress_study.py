#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermal_pinn.progress import run_progress_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-seed study for the preliminary report.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/progress"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_progress_study(args.output_dir, tuple(args.seeds), args.device)
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
