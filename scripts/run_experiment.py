#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermal_pinn import make_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adaptive thermal PINN experiment.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_config(args.mode, args.output_dir)
    cfg.device = args.device
    metrics = run_experiment(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
