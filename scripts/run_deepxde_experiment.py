#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermal_pinn import make_config
from thermal_pinn.deepxde_impl import run_deepxde_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an optional DeepXDE implementation of the thermal PINN.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deepxde_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_config(args.mode, args.output_dir)
    metrics = run_deepxde_experiment(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
