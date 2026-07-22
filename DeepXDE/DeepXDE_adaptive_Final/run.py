"""Run the validated 2D adaptive DeepXDE PINN from one TOML preset."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
WORKFLOW = HERE / "core" / "run_workflow.py"
ARCHITECTURES = HERE / "architectures"


def resolve_architecture(value: str) -> Path:
    supplied = Path(value)
    if supplied.suffix.lower() == ".toml" or supplied.parent != Path("."):
        candidate = supplied if supplied.is_absolute() else Path.cwd() / supplied
    else:
        candidate = ARCHITECTURES / f"{value}.toml"
    if not candidate.is_file():
        available = ", ".join(path.stem for path in sorted(ARCHITECTURES.glob("*.toml")))
        raise FileNotFoundError(
            f"Architecture file not found: {candidate}\nAvailable presets: {available}"
        )
    return candidate.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "architecture",
        nargs="?",
        default="balanced",
        help="preset name or path to a TOML architecture file (default: balanced)",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--paper-tables-dir", type=Path)
    args = parser.parse_args()

    architecture_path = resolve_architecture(args.architecture)
    command = [
        sys.executable,
        str(WORKFLOW),
        "--profile",
        args.profile,
        "--architecture-file",
        str(architecture_path),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    if args.paper_tables_dir is not None:
        command.extend(["--paper-tables-dir", str(args.paper_tables_dir.resolve())])

    print(f"Architecture: {architecture_path}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
