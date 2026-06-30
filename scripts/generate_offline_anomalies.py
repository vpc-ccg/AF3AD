"""Run the config-driven R3D-AD batch launcher from the repo root."""
from __future__ import annotations
import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the config-driven R3D-AD launcher over all configured categories."
    )
    parser.add_argument(
        "config",
        help="Path to the R3D-AD config file consumed by integrations.r3dad.train_test.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional run suffix forwarded to the launcher.",
    )
    args, passthrough = parser.parse_known_args()
    cmd = [
        sys.executable,
        "-m",
        "integrations.r3dad.train_test",
        args.config,
        "--tag",
        args.tag,
    ]
    cmd.extend(passthrough)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
