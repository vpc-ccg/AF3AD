"""Launch a meaningful PO3AD training run from the repo root."""
from __future__ import annotations
import argparse
import subprocess
import sys


COMMON_DEFAULTS = [
    "--save_freq", "500",
    "--optimizer", "AdamW",
    "--lr", "0.0008",
    "--lr_D", "0.0008",
    "--first_cycle_steps", "500",
    "--cycle_mult", "1.0",
    "--warmup_steps", "25",
    "--gamma", "0.2",
    "--data_repeat", "100",
    "--batch_size", "32",
    "--smart_anomaly",
    "--num_workers", "16",
    "--cache_dataset",
    "--cache_clear_freq", "10",
    "--gc_collect_freq", "5",
    "--metric_eval_freq", "500",
    "--validation_eval_freq", "500",
    "--voxel_size", "0.03",
    "--R_alpha", "2",
    "--R_beta", "2",
    "--R_low_bound", "0.03",
    "--R_up_bound", "0.25",
    "--B_alpha", "2",
    "--B_beta", "2",
    "--B_low_bound", "0.06",
    "--B_up_bound", "0.125",
    "--one_sided_prob", "0.5",
    "--intact_ratio", "0.1",
    "--no_plane_cut",
    "--offset_head_variant", "multi_head",
    "--offset_hidden_dim", "128",
    "--edge_cutout_enabled",
    "--mask_num", "64",
    "--sample_export_all",
    "--sample_export_annotated",
]

DATASET_DEFAULTS = {
    "AnomalyShapeNet": [
        "--epochs", "301",
    ],
    "Real3D": [
        "--epochs", "2501",
        "--lr_schedule", "cosine_warmup",
        "--max_lr", "0.0008",
        "--min_lr", "1e-06",
        "--downsample_mode", "voxel_fps",
        "--downsample_target_points", "15000",
        "--downsample_random_seed", "42",
        "--downsample_recompute_normals",
        "--validation_suffixes", "67,82,89,142,149,156,163,170,184,191",
        "--train_data_type", "cut",
    ],
}


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "integrations.po3ad.main",
        "--dataset", args.dataset,
        "--dataset_base_dir", args.dataset_base_dir,
        "--category", args.category,
        "--logpath", args.logpath,
        "--gpu_id", args.gpu_id,
    ]
    cmd.extend(COMMON_DEFAULTS)
    cmd.extend(DATASET_DEFAULTS[args.dataset])
    cmd.extend(passthrough)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the AF3AD online-synthesis PO3AD training pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DEFAULTS),
        required=True,
        help="Choose which PO3AD integration preset to run.",
    )
    parser.add_argument(
        "--dataset-base-dir",
        dest="dataset_base_dir",
        required=True,
        help="Root directory passed to the PO3AD dataset loader.",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Category or class name to train on.",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="CUDA device id string forwarded to PO3AD.",
    )
    parser.add_argument(
        "--logpath",
        default="./log/af3ad/",
        help="Directory used by the PO3AD trainer for checkpoints and logs.",
    )
    args, passthrough = parser.parse_known_args()
    subprocess.run(build_command(args, passthrough), check=True)


if __name__ == "__main__":
    main()
