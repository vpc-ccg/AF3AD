import os
import sys
import argparse
from pathlib import Path
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.r3dad.evaluation import ROC_AP
from integrations.r3dad.models.autoencoder import *
from integrations.r3dad.utils.data import *
from integrations.r3dad.utils.dataset import *
from integrations.r3dad.utils.misc import *
from integrations.r3dad.utils.transform import *


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='AutoEncoder')
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset', type=str, default='ShapeNetAD')
parser.add_argument('--dataset_path', type=str, default='./data/shapenet-ad')
parser.add_argument('--category', type=str, default='ashtray0')
parser.add_argument('--scale_mode', type=str, default=None)
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--num_aug', type=int, default=2048)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=128)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])
parser.add_argument('--rel', type=eval, default=False, choices=[True, False])
parser.add_argument('--use_patch', type=eval, default=False, choices=[True, False])
parser.add_argument('--patch_num', type=int, default=128)
parser.add_argument('--patch_scale', type=float, default=0.05)
parser.add_argument('--use_af3ad', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
parser.add_argument('--save_ply', type=eval, default=False, choices=[True, False])
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix=args.category + '_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
train_transforms = []
val_transforms = []
if args.rotate:
    train_transforms.append(RandomRotate(180, ['pointcloud']))
logger.info('Train Transforms: %s' % repr(train_transforms))
logger.info('Val Transforms: %s' % repr(val_transforms))
logger.info('Loading datasets...')
train_dset = getattr(sys.modules[__name__], args.dataset)(
    path=args.dataset_path,
    cates=[args.category],
    split='train',
    scale_mode=args.scale_mode,
    num_points=args.num_points,
    num_aug = args.num_aug,
    transforms=train_transforms,
    use_patch=args.use_patch,
    patch_num=args.patch_num,
    patch_scale=args.patch_scale,
    use_af3ad=args.use_af3ad,
)
val_dset = getattr(sys.modules[__name__], args.dataset)(
    path=args.dataset_path,
    cates=[args.category],
    split='test',
    scale_mode=args.scale_mode,
    num_points=args.num_points,
    transforms=val_transforms,
)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)

# Save preprocessed point clouds as PLY
if args.save_ply:
    ply_dir = os.path.join(log_dir if args.logging else '.', 'ply_preprocessed')
    logger.info('Saving preprocessed point clouds as PLY to %s ...' % ply_dir)
    train_dset.save_as_ply(os.path.join(ply_dir, 'train'))
    val_dset.save_as_ply(os.path.join(ply_dir, 'test'))
    logger.info('Done saving PLY files.')


# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = getattr(sys.modules[__name__], args.model)(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = getattr(sys.modules[__name__], args.model)(args).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

memory_bank = []
# Train, validate 
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)

    if it == 1:
        memory_bank.append(x)
        writer.add_mesh('train/pc', x, global_step=it)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    if args.rel:
        x_raw = batch['pointcloud_raw'].to(args.device)
        loss = model.get_loss(x, x_raw)
    else:
        loss = model.get_loss(x)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_loss(it):
    all_ref = []
    all_recons = []
    all_label = []
    all_mask = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
            if args.rel:
                recons += ref
        
        all_ref.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)
        all_label.append(batch['label'].to(args.device))
        all_mask.append(batch['mask'].to(args.device))

    all_ref = torch.cat(all_ref, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    all_label = torch.cat(all_label, dim=0)
    all_mask = torch.cat(all_mask, dim=0)

    metrics = ROC_AP(all_ref, all_recons, all_label, all_mask)
    roc_i, roc_p, ap_i, ap_p = metrics['ROC_i'].item(), metrics['ROC_p'].item(), metrics['AP_i'].item(), metrics['AP_p'].item()
    logger.info('[Val] Iter %04d | ROC_i_cdist %.6f | ROC_p_cdist %.6f | AP_i_cdist %.6f | AP_p_cdist %.6f' % (it, roc_i, roc_p, ap_i, ap_p))
    roc_i_nn, roc_p_nn, ap_i_nn, ap_p_nn = metrics['ROC_i_nn'].item(), metrics['ROC_p_nn'].item(), metrics['AP_i_nn'].item(), metrics['AP_p_nn'].item()
    logger.info('[Val] Iter %04d | ROC_i_nn %.6f | ROC_p_nn %.6f | AP_i_nn %.6f | AP_p_nn %.6f' % (it, roc_i_nn, roc_p_nn, ap_i_nn, ap_p_nn))
    writer.add_scalar('val/ROC_i', roc_i_nn, it)
    writer.add_scalar('val/ROC_p', roc_p_nn, it)
    writer.add_scalar('val/AP_i', ap_i_nn, it)
    writer.add_scalar('val/AP_p', ap_p_nn, it)
    writer.flush()

    np.save(os.path.join(log_dir, 'ref.npy'), all_ref.cpu().numpy())
    np.save(os.path.join(log_dir, 'out.npy'), all_recons.cpu().numpy())
    np.save(os.path.join(log_dir, 'mask.npy'), all_mask.cpu().numpy())

    return roc_i

def validate_inspect(it):
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch['pointcloud'].to(args.device)
        model.eval()
        code = model.encode(x)
        recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()
        if args.rel:
            recons += x

        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    writer.add_mesh('val/pc_in', x[:args.num_inspect_pointclouds], global_step=it)
    writer.add_mesh('val/pc_out', recons[:args.num_inspect_pointclouds], global_step=it)
    writer.flush()

# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                score = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, score, opt_states, it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
