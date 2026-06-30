import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pympler import muppy, summary
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from integrations.po3ad.configs.config_train import get_parser
from integrations.po3ad.models.discriminator import Discriminator
from integrations.po3ad.utils.log import get_logger
from integrations.po3ad.utils.memory_utils import clear_cuda_cache, report_memory_usage
from integrations.po3ad.utils.misc import (
    CosineAnnealingWarmupRestarts,
    _compute_epoch_metrics,
    _evaluate_training_pseudo_anomalies,
    _evaluate_validation_set,
    _format_metric,
    attach_file_handler,
    build_run_id,
    calculate_loss,
    cosine_lr_after_step,
    cosine_lr_with_warmup,
    create_optimizers,
    fix_seed,
    get_optimizer_lr,
    prepare_data,
    prepare_dirs,
    resume_from_checkpoint,
    save_anomaly_samples,
    save_anomaly_samples_wrapper,
    save_checkpoint,
    save_hparams,
)

PRESET_NAMES = [
    "Type 1: Basic Bulge",
    "Type 2: Basic Dent",
    "Type 3: Ridge",
    "Type 4: Trench",
    "Type 5: Elliptic Patch/Flat Spot",
    "Type 6: Skewed Impact Crater",
    "Type 7: Shear U",
    "Type 7b: Shear V",
    "Type 8: Double-sided Ripple",
    "Type 9: Micro Dimple Field Base",
    "Type 10: Directional Drag/Stretch",
]


# ------------------------------
# One-step D update
# ------------------------------


def print_top_memory_users(top_k=20):
    all_objects = muppy.get_objects()
    sum_obj = summary.summarize(all_objects)
    largest = sorted(sum_obj, key=lambda x: x[2], reverse=True)[:top_k]

    print("\n=== Top Memory Users ===")
    for obj in largest:
        obj_type, count, size = obj
        print(f"{obj_type:<40} | Count: {count:<8} | Size: {size/1024/1024:.4f} MB")


def configure_torch_runtime() -> None:
    """Set conservative PyTorch runtime defaults for reproducible training."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def configure_device(args) -> str:
    """Configure CUDA visibility and return the device string used by training."""
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = "cuda:0"
        torch.cuda.set_device(device)
        args.device = device
        return device
    args.device = "cpu"
    return "cpu"


def step_discriminator_lr(args, optimizer, lr_schedule, lr_scheduler, epoch: int) -> float:
    """Advance the discriminator LR schedule and return the current LR."""
    base_lr = args.lr_D if args.lr_D is not None else args.lr

    if lr_schedule == "cosine_warmup_restarts" and lr_scheduler is not None:
        lr_scheduler.step(epoch)
        return get_optimizer_lr(optimizer)

    if lr_schedule == "cosine_warmup":
        return cosine_lr_with_warmup(
            optimizer,
            base_lr,
            epoch,
            args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )

    cosine_lr_after_step(
        optimizer,
        base_lr,
        epoch,
        args.step_epoch,
        args.epochs,
        clip=args.min_lr,
    )
    return get_optimizer_lr(optimizer)


def log_metrics(writer: SummaryWriter, prefix: str, metrics: Dict[str, float], epoch: int) -> None:
    """Write finite scalar metrics to TensorBoard."""
    for metric_name, metric_value in metrics.items():
        if np.isfinite(metric_value):
            writer.add_scalar(f"{prefix}/{metric_name}", metric_value, epoch)


def run_periodic_maintenance(args, dataset, logger: logging.Logger, run_id: str, epoch: int) -> None:
    """Run optional memory, cache, and garbage-collection maintenance."""
    epoch_index = epoch + 1

    memory_report_freq = getattr(args, "memory_report_freq", 0)
    if memory_report_freq > 0 and epoch_index % memory_report_freq == 0:
        report_memory_usage(logger, epoch_index, prefix=f"[{run_id}]")
        clear_cuda_cache()

    gc_collect_freq = getattr(args, "gc_collect_freq", 50)
    if gc_collect_freq > 0 and epoch_index % gc_collect_freq == 0:
        clear_cuda_cache()
        collected = gc.collect()
        logger.info(
            "[%s] E%03d Aggressive GC | collected: %d objects",
            run_id,
            epoch_index,
            collected,
        )

    cache_clear_freq = getattr(args, "cache_clear_freq", 400)
    if cache_clear_freq <= 0 or epoch_index % cache_clear_freq != 0:
        return

    if hasattr(dataset, "clear_cache") and callable(getattr(dataset, "clear_cache")):
        cache_info = dataset.clear_cache(recreate=True)
        logger.info(
            "[%s] E%03d Cache %s | train: %d entries | test: %d entries",
            run_id,
            epoch_index,
            "recreated" if cache_info.get("recreated", False) else "cleared",
            cache_info["train_cache_cleared"],
            cache_info["test_cache_cleared"],
        )
        clear_cuda_cache()
        gc.collect()
        logger.info("[%s] E%03d Garbage collection completed", run_id, epoch_index)
        return

    logger.warning(
        "[%s] E%03d Cache clearing requested but dataset does not support clear_cache()",
        run_id,
        epoch_index,
    )


def log_dataset_presets(args, dataset, logger: logging.Logger) -> None:
    """Log anomaly preset metadata when the dataset exposes it."""
    if not (hasattr(dataset, "anomaly_presets") and hasattr(dataset, "num_presets")):
        return

    logger.info("Anomaly preset information:")
    logger.info("  Total presets: %d", dataset.num_presets)

    if hasattr(dataset, "preset_weights") and hasattr(dataset, "preset_probs"):
        logger.info("  Preset distribution weights and probabilities:")
        for idx, _ in enumerate(dataset.anomaly_presets):
            name = PRESET_NAMES[idx] if idx < len(PRESET_NAMES) else f"Preset {idx}"
            weight = dataset.preset_weights[idx] if idx < len(dataset.preset_weights) else 1.0
            prob = dataset.preset_probs[idx] if idx < len(dataset.preset_probs) else 1.0 / dataset.num_presets
            logger.info(
                "    Preset %d (%s): weight=%.2f, prob=%.4f",
                idx,
                name,
                weight,
                prob,
            )
    else:
        for idx, _ in enumerate(dataset.anomaly_presets):
            if idx < len(PRESET_NAMES):
                logger.info("  Preset %d: %s", idx, PRESET_NAMES[idx])

    logger.info("  Hyperparameter ranges:")
    logger.info(
        "    R (radius): [%s, %s] with Beta(%s, %s)",
        args.R_low_bound,
        args.R_up_bound,
        args.R_alpha,
        args.R_beta,
    )
    logger.info(
        "    B (magnitude): [%s, %s] with Beta(%s, %s)",
        args.B_low_bound,
        args.B_up_bound,
        args.B_alpha,
        args.B_beta,
    )


def discriminator_step(
    args,
    discriminator: nn.Module,
    raw_batch: Dict[str, Any],
    opt_D: optim.Optimizer,
    scaler_D: GradScaler,
    amp_enabled: bool,
):
    device = next(discriminator.parameters()).device
    with autocast(enabled=amp_enabled):
        # model should move tensors as needed
        pred_offset = discriminator(raw_batch)
        gt_offset = raw_batch['batch_offset'].to(device, non_blocking=True)

        # Get point coordinates for smoothness regularization if enabled
        point_coords = None
        if getattr(args, 'lambda_smooth_reg', 0.0) > 0:
            # Use batch_xyz if available for smoothness computation
            point_coords = raw_batch.get('batch_xyz')
            if point_coords is not None:
                point_coords = point_coords.to(device, non_blocking=True)

        loss_D, _, _, _, offset_norm_loss, offset_dir_loss = calculate_loss(
            gt_offset,
            pred_offset,
            loss_variant=getattr(args, 'loss_variant', 'baseline'),
            focal_alpha=getattr(args, 'focal_alpha', 0.25),
            focal_gamma=getattr(args, 'focal_gamma', 2.0),
            focal_tau=getattr(args, 'focal_tau', 0.01),
            lambda_aux_focal=getattr(args, 'lambda_aux_focal', 0.1),
            # Regularization parameters
            lambda_l1_reg=getattr(args, 'lambda_l1_reg', 0.0),
            lambda_l2_reg=getattr(args, 'lambda_l2_reg', 0.0),
            lambda_smooth_reg=getattr(args, 'lambda_smooth_reg', 0.0),
            edge_aware_weight=getattr(args, 'edge_aware_weight', 0.0),
            point_coords=point_coords,
        )

    opt_D.zero_grad(set_to_none=True)
    scaler_D.scale(loss_D).backward()

    # FIX 4: ADD GRADIENT CLIPPING FOR THE DISCRIMINATOR
    # This prevents the discriminator's weights from exploding when it sees
    # a particularly effective anomaly from the generator.
    scaler_D.unscale_(opt_D)
    nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)

    scaler_D.step(opt_D)
    scaler_D.update()
    return float(loss_D.detach().cpu()), float(offset_norm_loss.detach().cpu()), float(offset_dir_loss.detach().cpu())


def train(
    args,
    standard_loader,
    train_random_data_loader,
    rollout_loader,
    val_loader,
    dataset,
    run_id: str,
    run_dir: Path,
    logs_dir: Path,
    ckpt_dir: Path,
    logger: logging.Logger,
    writer: SummaryWriter
):
    """Train the policy optimizer and anomaly detector in a co-evolutionary manner.

    Args:
        args (_type_): _description_
        standard_loader (_type_): _description_
        rollout_loader (_type_): _description_
        dataset (_type_): _description_
        run_id (str): _description_
        run_dir (Path): _description_
        logs_dir (Path): _description_
        ckpt_dir (Path): _description_
        logger (logging.Logger): _description_
        writer (SummaryWriter): _description_

    Raises:
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """

    # Set general variables
    device = getattr(args, "device", f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    validation_eval_freq = max(
        0, int(getattr(args, 'validation_eval_freq', 0)))
    validation_enabled = validation_eval_freq > 0 and val_loader is not None
    if validation_eval_freq > 0 and not validation_enabled:
        logger.warning(
            "Validation evaluation requested but no validation samples were found.")


    # Build discriminator or anomaly detection model (default: PO3AD)
    discriminator = Discriminator(args).to(device)


    # Prepare the directory for saving anomaly synthesis samples
    samples_root = run_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    # Build Optimizers
    # Update
    opt_D = create_optimizers(args, discriminator)

    # Build AMP and its scaler
    amp_enabled = torch.cuda.is_available()
    scaler_D = GradScaler(enabled=amp_enabled)

    # Define LR schedulers
    #   cosine_after_step: decrase with cosine after step (default)
    #   cosine_warmup_restarts: a more sophisticated cosine annealing with warmup and preiodical restarts
    #        (Adapted from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup)
    lr_schedule = getattr(args, 'lr_schedule', 'cosine_after_step')
    lr_scheduler_D = None
    if lr_schedule == 'cosine_warmup_restarts':
        max_lr = args.max_lr if args.max_lr is not None else (
            args.lr_D if args.lr_D is not None else args.lr
        )
        lr_scheduler_D = CosineAnnealingWarmupRestarts(
            opt_D,
            first_cycle_steps=args.first_cycle_steps,
            cycle_mult=args.cycle_mult,
            max_lr=max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            gamma=args.gamma,
        )

    # Dataloader iterators
    standard_iter = iter(standard_loader)

    # Variables for tracking best model
    start_epoch = 0
    best_d_loss = float("inf")
    best_epoch = -1

    # Update
    # Set resume checkpoints if specified
    start_epoch, best_d_loss, best_epoch = resume_from_checkpoint(
        args, run_id, logger, device,
        discriminator, opt_D,
        scaler_D
    )


    # In-class function to fetch standard batches for discriminator updates
    def fetch_standard_batch(params: Dict[str, Any]):
        """Fetch a batch from the standard dataloader, after putting params into the queue.

        Args:
            params (Dict[str, Any]): _description_

        Returns:
            _type_: _description_
        """
        nonlocal standard_iter
        dataset.standard_param_queue.put(params)

        try:
            return next(standard_iter)
        except StopIteration:
            standard_iter = iter(standard_loader)
            return next(standard_iter)

    # Prepare the training loop
    num_iters_per_epoch = len(standard_loader)
    D_loss_epoch = float("nan")


    # Epoch loop
    for epoch in range(start_epoch, args.epochs):
        d_meter, p_meter, offset_norm_loss_meter, offset_dir_loss_meter = [], [], [], []

        saved_samples_this_epoch = False

        current_lr_D = (
            step_discriminator_lr(args, opt_D, lr_schedule, lr_scheduler_D, epoch)
            if opt_D is not None
            else 0.0
        )

        # Determine if we should collect metrics this epoch for saving it and logging
        collect_metrics = (
            getattr(args, 'metric_eval_freq', 0) > 0
            and ((epoch + 1) % getattr(args, 'metric_eval_freq', 0) == 0)
        )
        object_scores_epoch: List[float] = []
        object_labels_epoch: List[int] = []
        point_scores_epoch: List[np.ndarray] = []
        point_labels_epoch: List[np.ndarray] = []
        points_collected = 0
        max_points = getattr(args, 'metric_max_points', None)
        if max_points is not None:
            max_points = int(max_points)
            if max_points <= 0:
                max_points = None


        # Iteration loop
        for it in range(num_iters_per_epoch):
            # ============================================================================
            # RL CO-TRAINING LOOP
            # ============================================================================
            # This loop implements co-training between:
            #   - Discriminator: Learns to detect anomalies in point clouds
            #
            # Training alternates between:
            #   5. Update discriminator on best anomaly to minimize error
            # ============================================================================
            


            # -------- 5. Train the Discriminator on the psudeo anomaly samples --------

            batch = fetch_standard_batch({
                'collate_mode': 'random',
            })

            # -------- Save anomaly samples (outside training loop, once per epoch) --------
            # Generate samples for visualization/analysis regardless of RL training mode
            saved_samples_this_epoch = save_anomaly_samples_wrapper(
                args=args,
                epoch=epoch,
                it=0,
                saved_samples_this_epoch=saved_samples_this_epoch,
                data_samples=batch,
                samples_root=samples_root,
                logger=logger,
                save_fn=save_anomaly_samples,
            )

            # base_lr = args.lr_D if args.lr_D is not None else args.lr
            # cosine_lr_after_step(opt_D, base_lr, epoch,
            #                     args.step_epoch, args.epochs, clip=1e-6)
            d_loss_val, offset_norm_loss, offset_dir_loss = discriminator_step(
                args, discriminator, batch, opt_D, scaler_D, amp_enabled)
            d_meter.append(d_loss_val)
            offset_norm_loss_meter.append(offset_norm_loss)
            offset_dir_loss_meter.append(offset_dir_loss)

        # ----- Logging per epoch -----
        D_loss_epoch = float(np.mean(d_meter)) if len(
            d_meter) else float("nan")

        offset_norm_loss_epoch = float(np.mean(offset_norm_loss_meter)) if len(
            d_meter) else float("nan")
        offset_dir_loss_epoch = float(np.mean(offset_dir_loss_meter)) if len(
            d_meter) else float("nan")


        logger.info(
            f"[{run_id}] E{epoch:03d} D_loss={D_loss_epoch:.4f}")
        writer.add_scalar("loss/D_epoch", D_loss_epoch, epoch)
        writer.add_scalar("loss/Offset_norm_loss_epoch",
                          offset_norm_loss_epoch, epoch)
        writer.add_scalar("loss/Offset_dir_loss_epoch",
                          offset_dir_loss_epoch, epoch)
        writer.add_scalar("lr/D", current_lr_D, epoch)


        if collect_metrics:
            metrics = _compute_epoch_metrics(
                object_scores_epoch,
                object_labels_epoch,
                point_scores_epoch,
                point_labels_epoch,
            )
            logger.info(
                "[%s] E%03d metrics | obj AUC-ROC=%s | obj AP=%s | point AUC-ROC=%s | point AP=%s",
                run_id,
                epoch,
                _format_metric(metrics['object_auc_roc']),
                _format_metric(metrics['object_auc_pr']),
                _format_metric(metrics['point_auc_roc']),
                _format_metric(metrics['point_auc_pr']),
            )

            log_metrics(writer, "metrics", metrics, epoch)

        # Evaluation on validation set
        if validation_enabled and ((epoch + 1) % validation_eval_freq == 0):
            val_metrics = _evaluate_validation_set(
                args,
                discriminator,
                dataset,
                val_loader,
            )
            if val_metrics is not None:
                logger.info(
                    "[%s] E%03d validation | obj AUC-ROC=%s | obj AP=%s | point AUC-ROC=%s | point AP=%s",
                    run_id,
                    epoch,
                    _format_metric(val_metrics['object_auc_roc']),
                    _format_metric(val_metrics['object_auc_pr']),
                    _format_metric(val_metrics['point_auc_roc']),
                    _format_metric(val_metrics['point_auc_pr']),
                )

                log_metrics(writer, "val", val_metrics, epoch)


        # Training evaluation on pseudo anomalous samples (overfitting check)
        train_eval_freq = getattr(args, 'train_eval_freq', 0)
        if train_eval_freq > 0 and ((epoch + 1) % train_eval_freq == 0):
            num_batches = getattr(args, 'train_eval_num_batches', 5)
            train_eval_metrics = _evaluate_training_pseudo_anomalies(
                args,
                discriminator,
                rollout_loader,
                dataset,
                num_batches=num_batches,
            )
            if train_eval_metrics is not None:
                logger.info(
                    "[%s] E%03d train_eval (overfitting check) | obj AUC-ROC=%s | obj AP=%s | point AUC-ROC=%s | point AP=%s",
                    run_id,
                    epoch,
                    _format_metric(train_eval_metrics['object_auc_roc']),
                    _format_metric(train_eval_metrics['object_auc_pr']),
                    _format_metric(train_eval_metrics['point_auc_roc']),
                    _format_metric(train_eval_metrics['point_auc_pr']),
                )

                log_metrics(writer, "train_eval", train_eval_metrics, epoch)

        run_periodic_maintenance(args, dataset, logger, run_id, epoch)

        if np.isfinite(D_loss_epoch) and D_loss_epoch < best_d_loss:
            best_d_loss = D_loss_epoch
            best_epoch = epoch + 1
            save_checkpoint(
                ckpt_dir,
                run_id,
                tag="best",
                discriminator=discriminator,
                opt_D=opt_D,
                epoch=epoch + 1,
                scaler_D=scaler_D,
                metrics={"d_loss": D_loss_epoch},
            )
            logger.info(
                "[%s] New best discriminator loss %.4f at epoch %03d; checkpoint saved.",
                run_id,
                best_d_loss,
                best_epoch,
            )

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(ckpt_dir, run_id, tag="epoch",
                            discriminator=discriminator,
                            opt_D=opt_D, epoch=epoch+1,
                            scaler_D=scaler_D,
                            metrics={
                                "d_loss": D_loss_epoch,
                                "best_d_loss": best_d_loss,
                                "best_epoch": best_epoch,
                            })

    if best_epoch != -1:
        logger.info(
            "[%s] Best discriminator loss %.4f achieved at epoch %03d",
            run_id,
            best_d_loss,
            best_epoch,
        )
    else:
        logger.warning(
            "[%s] No finite discriminator loss recorded; best checkpoint not saved.", run_id)

    save_checkpoint(ckpt_dir, run_id, tag="final",
                    discriminator=discriminator,
                    opt_D=opt_D,  epoch=args.epochs,
                    scaler_D=scaler_D, 
                    metrics={
                        "d_loss": D_loss_epoch,
                        "best_d_loss": best_d_loss,
                        "best_epoch": best_epoch,
                    })


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    configure_torch_runtime()
    args = get_parser()
    configure_device(args)

    # Fix seed
    fix_seed(args.manual_seed)

    # Build run ID and prepare directories
    run_id = build_run_id(args)
    run_dir, logs_dir, ckpt_dir = prepare_dirs(args, run_id)

    # Build logger
    logger = get_logger(args)
    attach_file_handler(logger, logs_dir / "train.log")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(args)

    # Save hyperparameters and initialize TensorBoard writer
    save_hparams(run_dir, args)
    writer = SummaryWriter(str(logs_dir))

    # Preapre dataset and dataloaders: standard one for D update, rollout one for P update
    dataset, standard_loader, train_random_data_loader, rollout_loader, val_loader = prepare_data(
        args)

    log_dataset_presets(args, dataset, logger)

    # Start training
    train(args=args,
          standard_loader=standard_loader,
          train_random_data_loader=train_random_data_loader,
          rollout_loader=rollout_loader,
          val_loader=val_loader,
          dataset=dataset,
          run_id=run_id,
          run_dir=run_dir,
          logs_dir=logs_dir,
          ckpt_dir=ckpt_dir,
          logger=logger,
          writer=writer)

    # Close TensorBoard writer
    writer.close()
