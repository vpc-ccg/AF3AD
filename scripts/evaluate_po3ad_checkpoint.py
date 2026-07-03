"""Evaluate a PO3AD-style AF3AD checkpoint from a YAML config."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = None
torch = None


EVAL_DEFAULTS: Dict[str, Any] = {
    "task": "eval",
    "manual_seed": 42,
    "gpu_id": "0",
    "dataset": "AnomalyShapeNet",
    "dataset_base_dir": "data/AnomalyShapeNet/dataset",
    "category": "ashtray0",
    "batch_size": 1,
    "rollout_batch_size": 1,
    "data_repeat": 1,
    "mask_num": 64,
    "num_works": 4,
    "num_workers": 4,
    "cache_dataset": False,
    "cache_test_set": False,
    "validation": False,
    "validation_suffixes": "",
    "voxel_size": 0.03,
    "in_channels": 3,
    "out_channels": 32,
    "offset_head_variant": "baseline",
    "offset_hidden_dim": 64,
    "offset_num_layers": 3,
    "offset_dropout": 0.0,
    "offset_attention_reduction": 4,
    "train_data_type": "pcd",
    "downsample_mode": "none",
    "downsample_ratio": 0.4,
    "downsample_voxel_size": None,
    "downsample_voxel_size_multiplier": 2.0,
    "downsample_target_points": None,
    "downsample_recompute_normals": True,
    "downsample_random_seed": 42,
    "test_downsample_enabled": False,
    "test_downsample_voxel_size": 0.001,
    "plane_cut_enabled": False,
    "edge_cutout_enabled": False,
    "smart_anomaly": True,
    "synthesis_policy": "af3ad_presets",
    "af3ad_policy": "plain",
    "preset_randomfactory_prob": 0.5,
    "binary_anomaly_label": False,
    "intact_ratio": 0.0,
    "R_alpha": 2.0,
    "R_beta": 2.0,
    "R_low_bound": 0.03,
    "R_up_bound": 0.25,
    "B_alpha": 2.0,
    "B_beta": 2.0,
    "B_low_bound": 0.06,
    "B_up_bound": 0.125,
    "one_sided_prob": 0.5,
    "poly_q": 4.0,
    "p_bulge": 0.5,
    "metric_max_points": 0,
    "max_samples": None,
    "strict_checkpoint": True,
    "save_sample_scores": True,
    "save_point_scores": False,
    "verbose_samples": False,
    "write_metrics_json": True,
    "logpath": "./log/po3ad_eval",
    "output_dir": "",
}


NESTED_CONFIG_SECTIONS = {
    "model",
    "data",
    "dataloader",
    "evaluation",
    "eval",
    "logging",
    "runtime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PO3AD-style AF3AD discriminator checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a PO3AD-style checkpoint containing discriminator weights.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML evaluation config.",
    )
    parser.add_argument("--dataset-base-dir", dest="dataset_base_dir")
    parser.add_argument("--category")
    parser.add_argument("--gpu-id", dest="gpu_id")
    parser.add_argument("--voxel-size", dest="voxel_size", type=float)
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--max-samples", dest="max_samples", type=int)
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument(
        "--verbose-samples",
        action="store_true",
        help="Print each evaluated sample path as it is processed.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any resolved config value. Values are parsed as YAML scalars.",
    )
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global np, torch

    try:
        import numpy as numpy_module
        import torch as torch_module
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PO3AD evaluation requires NumPy and PyTorch. Activate the PO3AD "
            "environment before running this script."
        ) from exc

    np = numpy_module
    torch = torch_module


def load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PyYAML is required to read evaluation configs. Install it with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(loaded).__name__}.")
    return loaded


def parse_override_value(value: str) -> Any:
    try:
        import yaml
    except ImportError:
        return value
    parsed = yaml.safe_load(value)
    return value if parsed is None and value.strip().lower() != "null" else parsed


def merge_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(EVAL_DEFAULTS)
    for key, value in raw_config.items():
        if key == "dataset" and isinstance(value, dict):
            if "name" in value:
                config["dataset"] = value["name"]
            if "base_dir" in value:
                config["dataset_base_dir"] = value["base_dir"]
            if "dataset_base_dir" in value:
                config["dataset_base_dir"] = value["dataset_base_dir"]
            if "category" in value:
                config["category"] = value["category"]
            for nested_key, nested_value in value.items():
                if nested_key not in {"name", "base_dir", "dataset_base_dir", "category"}:
                    config[nested_key] = nested_value
        elif key in NESTED_CONFIG_SECTIONS and isinstance(value, dict):
            config.update(value)
        else:
            config[key] = value

    if "num_workers" in config and config.get("num_workers") is not None:
        config["num_works"] = int(config["num_workers"])
    elif "num_works" in config and config.get("num_works") is not None:
        config["num_workers"] = int(config["num_works"])

    config["task"] = "eval"
    return config


def apply_cli_overrides(config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    for key in (
        "dataset_base_dir",
        "category",
        "gpu_id",
        "voxel_size",
        "num_workers",
        "max_samples",
        "output_dir",
    ):
        value = getattr(cli_args, key, None)
        if value is not None:
            config[key] = value

    for item in cli_args.overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set expects a non-empty key, got: {item}")
        config[key] = parse_override_value(value)

    if "num_workers" in config and config.get("num_workers") is not None:
        config["num_works"] = int(config["num_workers"])
    if bool(getattr(cli_args, "verbose_samples", False)):
        config["verbose_samples"] = True
    return config


def namespace_from_config(config: Dict[str, Any]) -> SimpleNamespace:
    args = SimpleNamespace(**config)
    args.num_workers = int(getattr(args, "num_workers", getattr(args, "num_works", 0)))
    args.num_works = int(getattr(args, "num_works", args.num_workers))
    return args


def configure_torch_runtime(num_threads: int = 1) -> None:
    torch.set_num_threads(max(1, int(num_threads)))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def configure_cuda(args: SimpleNamespace) -> str:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PO3AD evaluation requires CUDA in this integration because the discriminator "
            "constructs MinkowskiEngine SparseTensor objects on CUDA."
        )
    device = "cuda:0"
    torch.cuda.set_device(device)
    args.device = device
    return device


def build_output_dir(config: Dict[str, Any], checkpoint_path: Path) -> Path:
    explicit = str(config.get("output_dir") or "").strip()
    if explicit:
        output_dir = Path(explicit)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        dataset = sanitize_for_path(str(config.get("dataset", "dataset")))
        category = sanitize_for_path(str(config.get("category", "category")))
        ckpt = sanitize_for_path(checkpoint_path.stem)
        output_dir = Path(str(config.get("logpath") or "./log/po3ad_eval")) / (
            f"{dataset}_{category}_{ckpt}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_for_path(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "item"


def load_checkpoint_state(
    checkpoint_path: Path,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    metadata: Dict[str, Any] = {}

    if isinstance(checkpoint, dict):
        metadata = checkpoint_metadata(checkpoint)
        for key in ("discriminator", "model", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return strip_module_prefix(state), metadata

        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return strip_module_prefix(checkpoint), metadata

    raise ValueError(
        "Unsupported checkpoint format. Expected a dict containing "
        "`discriminator`, `model`, `state_dict`, or a raw state_dict."
    )


def checkpoint_metadata(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Keep lightweight checkpoint metadata and skip optimizer/scaler payloads."""

    metadata: Dict[str, Any] = {}
    skipped = {
        "discriminator",
        "model",
        "state_dict",
        "opt_D",
        "optimizer",
        "scaler_D",
        "scheduler",
    }
    for key, value in checkpoint.items():
        if key in skipped:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        elif isinstance(value, dict) and key == "metrics":
            metadata[key] = value
        else:
            metadata[key] = f"<{type(value).__name__}>"
    return metadata


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        stripped[key[7:] if key.startswith("module.") else key] = value
    return stripped


def gt_mask_for_batch(dataset: Any, batch: Dict[str, Any], pred_mask: np.ndarray) -> np.ndarray:
    if "gt_masks" in batch:
        gt_mask = tensor_to_numpy(batch["gt_masks"]).astype(np.float32).reshape(-1)
    else:
        sample_name = Path(batch["fn"][0]).stem
        normal_tag = getattr(dataset, "normal_tag", None)
        if normal_tag and normal_tag in sample_name:
            gt_mask = np.zeros(pred_mask.shape[0], dtype=np.float32)
        else:
            gt_path = dataset._resolve_gt_path(sample_name)
            kwargs: Dict[str, Any] = {}
            delimiter = getattr(dataset, "gt_delimiter", ",")
            if delimiter is not None:
                kwargs["delimiter"] = delimiter
            gt_data = np.loadtxt(gt_path, **kwargs)
            if gt_data.ndim == 1:
                gt_data = gt_data.reshape(1, -1)
            if gt_data.shape[1] < 4:
                raise ValueError(f"Ground-truth file has fewer than 4 columns: {gt_path}")
            gt_mask = gt_data[:, 3].astype(np.float32)

    if gt_mask.shape[0] != pred_mask.shape[0]:
        sample = Path(batch["fn"][0]).name
        raise ValueError(
            f"Prediction/ground-truth length mismatch for {sample}: "
            f"{pred_mask.shape[0]} predictions vs {gt_mask.shape[0]} labels."
        )
    return gt_mask


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_point_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    point_scores: List[np.ndarray],
    point_labels: List[np.ndarray],
    points_collected: int,
    max_points: Optional[int],
) -> int:
    if max_points is not None and points_collected >= max_points:
        return points_collected

    pred_local = pred_mask
    gt_local = gt_mask
    if max_points is not None:
        remaining = max_points - points_collected
        sample_size = min(remaining, pred_local.shape[0])
        if sample_size <= 0:
            return points_collected
        if sample_size < pred_local.shape[0]:
            indices = np.random.choice(pred_local.shape[0], sample_size, replace=False)
            pred_local = pred_local[indices]
            gt_local = gt_local[indices]

    point_scores.append(pred_local)
    point_labels.append(gt_local)
    return points_collected + pred_local.shape[0]


def write_sample_scores(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_point_score_npz(output_dir: Path, sample_name: str, pred: np.ndarray, gt: np.ndarray) -> None:
    point_dir = output_dir / "point_scores"
    point_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        point_dir / f"{sanitize_for_path(Path(sample_name).stem)}.npz",
        point_scores=pred.astype(np.float32),
        point_labels=gt.astype(np.float32),
    )


def evaluate(args: SimpleNamespace, checkpoint_path: Path, output_dir: Path) -> Dict[str, float]:
    from integrations.po3ad.data.factory import get_dataset_class
    from integrations.po3ad.models.discriminator import Discriminator
    from integrations.po3ad.utils.misc import _compute_epoch_metrics, _eval_batch, fix_seed

    fix_seed(int(args.manual_seed))

    dataset_cls = get_dataset_class(args.dataset)
    dataset = dataset_cls(args)
    dataset.testLoader()
    test_loader = getattr(dataset, "test_data_loader", None)
    if test_loader is None:
        raise RuntimeError("Dataset did not create a test_data_loader.")
    if len(test_loader) == 0:
        raise RuntimeError(
            f"No test samples found for dataset={args.dataset}, category={args.category}."
        )

    model = Discriminator(args).to(args.device)
    state_dict, checkpoint_metadata = load_checkpoint_state(checkpoint_path, args.device)
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=bool(getattr(args, "strict_checkpoint", True)),
    )
    if missing or unexpected:
        print(
            "Checkpoint key mismatch:",
            json.dumps({"missing": list(missing), "unexpected": list(unexpected)}, indent=2),
            file=sys.stderr,
        )
    model.eval()

    max_samples = getattr(args, "max_samples", None)
    max_samples = int(max_samples) if max_samples not in (None, "", 0) else None
    metric_max_points = int(getattr(args, "metric_max_points", 0) or 0)
    max_points = metric_max_points if metric_max_points > 0 else None

    object_scores: List[float] = []
    object_labels: List[int] = []
    point_scores: List[np.ndarray] = []
    point_labels: List[np.ndarray] = []
    sample_rows: List[Dict[str, Any]] = []
    points_collected = 0

    with torch.no_grad():
        for sample_index, batch in enumerate(test_loader):
            if max_samples is not None and sample_index >= max_samples:
                break

            if bool(getattr(args, "verbose_samples", False)):
                print(sample_index, batch["fn"][0])
            torch.cuda.empty_cache()
            sample_score, pred_offset = _eval_batch(batch, model)
            pred_mask = pred_offset.detach().cpu().abs().sum(dim=-1).numpy()
            gt_mask = gt_mask_for_batch(dataset, batch, pred_mask)
            object_label = int(tensor_to_numpy(batch["labels"]).reshape(-1)[0])
            object_score = float(sample_score.item())
            sample_name = str(batch["fn"][0])

            object_scores.append(object_score)
            object_labels.append(object_label)
            points_collected = add_point_metrics(
                pred_mask,
                gt_mask,
                point_scores,
                point_labels,
                points_collected,
                max_points,
            )

            sample_rows.append(
                {
                    "index": sample_index,
                    "file": sample_name,
                    "label": object_label,
                    "object_score": object_score,
                    "point_count": int(pred_mask.shape[0]),
                    "point_score_mean": float(pred_mask.mean()) if pred_mask.size else 0.0,
                    "point_score_max": float(pred_mask.max()) if pred_mask.size else 0.0,
                    "gt_positive_points": int((gt_mask > 0.5).sum()),
                }
            )

            if bool(getattr(args, "save_point_scores", False)):
                save_point_score_npz(output_dir, sample_name, pred_mask, gt_mask)

    metrics = _compute_epoch_metrics(
        object_scores,
        object_labels,
        point_scores,
        point_labels,
    )
    metrics.update(
        {
            "num_samples": float(len(object_scores)),
            "num_anomalous_samples": float(sum(object_labels)),
            "num_point_labels_used": float(points_collected),
        }
    )

    if bool(getattr(args, "save_sample_scores", True)):
        write_sample_scores(output_dir / "sample_scores.csv", sample_rows)

    if bool(getattr(args, "write_metrics_json", True)):
        payload = {
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "config": vars(args),
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    return metrics


def print_metrics(metrics: Dict[str, float], output_dir: Path) -> None:
    print("PO3AD evaluation complete")
    print(f"  output_dir: {output_dir}")
    for key in (
        "num_samples",
        "num_anomalous_samples",
        "num_point_labels_used",
        "object_auc_roc",
        "object_auc_pr",
        "point_auc_roc",
        "point_auc_pr",
    ):
        value = metrics.get(key)
        if isinstance(value, float) and np.isfinite(value):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    cli_args = parse_args()
    checkpoint_path = Path(cli_args.checkpoint)
    config_path = Path(cli_args.config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw_config = load_yaml_config(config_path)
    config = apply_cli_overrides(merge_config(raw_config), cli_args)
    args = namespace_from_config(config)

    load_runtime_dependencies()
    configure_torch_runtime(int(getattr(args, "torch_num_threads", 1)))
    configure_cuda(args)
    output_dir = build_output_dir(config, checkpoint_path)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    metrics = evaluate(args, checkpoint_path, output_dir)
    print_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
