"""Launch a meaningful AF3AD-backed R3D-AD training run from the repo root."""
from __future__ import annotations
import argparse
import subprocess
import sys


DEFAULT_ARGS = [
    "--dataset", "ShapeNetAD",
    "--num_points", "15000",
    "--num_aug", "2048",
    "--train_batch_size", "32",
    "--val_batch_size", "32",
    "--max_iters", "40000",
    "--val_freq", "10000",
    "--seed", "42",
    "--save_ply", "True",
    "--use_af3ad", "True",
    "--tag", "af3ad",
]


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "integrations.r3dad.train_ae",
        "--dataset_path", args.dataset_path,
        "--category", args.category,
        "--log_root", args.log_root,
    ]
    cmd.extend(DEFAULT_ARGS)
    cmd.extend(passthrough)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the AF3AD reference run for the R3D-AD integration."
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        required=True,
        help="Path to the ShapeNetAD point-cloud directory.",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="ShapeNetAD category to train on.",
    )
    parser.add_argument(
        "--log-root",
        default="./logs_ae/comparison_af3ad",
        help="Base log directory for the run.",
    )
    args, passthrough = parser.parse_known_args()
    subprocess.run(build_command(args, passthrough), check=True)


if __name__ == "__main__":
    main()
