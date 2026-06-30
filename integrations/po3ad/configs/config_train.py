import argparse

# config para


def get_parser():
    """Prepare the parser for training

    Returns:
        argparse.ArgumentParser: parser with training parameters
    """

    # # Train config
    parser = argparse.ArgumentParser(description='3D anomaly detection')
    parser.add_argument('--task', type=str, default='train',
                        help='task: train or test')
    parser.add_argument('--manual_seed', type=int,
                        default=42, help='seed to produce')
    parser.add_argument('--epochs', type=int, default=1001, help='Total epoch')
    parser.add_argument('--num_works', '--num_workers', dest='num_works', type=int, default=16,
                        help='Number of dataset worker processes')
    parser.add_argument('--pretrain', type=str, default='',
                        help='path to pretrain model')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='Pre-training model saving frequency(epoch)')
    parser.add_argument('--logpath', type=str,
                        default='./log/ashtray0/', help='path to save logs')
    parser.add_argument('--validation', type=bool, default=True,
                        help='Whether to verify the validation set')
    parser.add_argument('--validation_eval_freq', type=int, default=0,
                        help='Run validation metrics every N epochs (0 to disable)')
    parser.add_argument('--validation_suffixes', type=str, default='',
                        help='Comma-separated suffix indices to build the validation split (leave empty for automatic lowest-two selection)')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Optional experiment name for logging and checkpoints')
    parser.add_argument('--loss_variant', type=str, default='baseline', choices=['baseline', 'focal_cls', 'focal_reg'],
                        help='Select baseline, focal classification, or focal-weighted regression loss variant')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha balancing factor for focal classification loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma focusing parameter for focal loss variants')
    parser.add_argument('--focal_tau', type=float, default=0.01,
                        help='Tau parameter for focal regression weighting')
    parser.add_argument('--lambda_aux_focal', type=float, default=0.1,
                        help='Scaling for auxiliary focal classification loss')
    
    # Regularization parameters for offset prediction
    parser.add_argument('--lambda_l1_reg', type=float, default=0.0,
                        help='L1 regularization weight for predicted offsets (sparsity, 0.0 to disable)')
    parser.add_argument('--lambda_l2_reg', type=float, default=0.0,
                        help='L2 regularization weight for predicted offsets (smoothness, 0.0 to disable)')
    parser.add_argument('--lambda_smooth_reg', type=float, default=0.0,
                        help='Smoothness regularization weight (spatial consistency, 0.0 to disable)')
    parser.add_argument('--edge_aware_weight', type=float, default=0.0,
                        help='Edge-aware loss weighting factor (0.0-1.0, downweights edge points, 0.0 to disable)')

    # #Dataset setting
    parser.add_argument('--dataset', type=str,
                        default='AnomalyShapeNet', help='datasets', choices=['AnomalyShapeNet', 'Real3D'])
    # parser.add_argument('--dataset_base_dir', type=str,
    #                     default='data/AnomalyShapeNet/dataset', help='base path to dataset')
    parser.add_argument('--dataset_base_dir', type=str,
                        default='data/Real3D', help='base path to dataset')
    parser.add_argument('--category', type=str,
                        default='ashtray0', help='categories for each class')
    parser.add_argument('--train_data_type', type=str, default='cut',
                        choices=['pcd', 'cut', 'cut_full'],
                        help='Training data type for Real3D: pcd (Real3D-AD-PCD), '
                             'cut (Real3D-AD-PLY-CUT), cut_full (Real3D-AD-PLY-CUT+FULL)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size for single GPU')
    parser.add_argument('--rollout_batch_size', type=int, default=1,
                        help='batch size for rollout dataloader')
    parser.add_argument('--data_repeat', type=int, default=100,
                        help='repeat the data for each epoch')
    parser.add_argument('--mask_num', type=int, default=64)
    parser.add_argument('--cache_dataset', action='store_true', default=False,
                        help='cache training meshes in memory to speed up epochs')
    parser.add_argument('--cache_test_set', action='store_true', default=None,
                        help='cache test point clouds (defaults to cache_dataset when unset)')
    
    # TODO: Remove 
    parser.add_argument('--cache_stats_freq', type=int, default=1,
                        help='frequency of cache statistics logging (epochs, 0=disabled)')
    parser.add_argument('--worker_cache_log_interval', type=int, default=100,
                        help='log worker cache stats every N file loads (0=disabled)')

    # Point cloud downsampling parameters
    parser.add_argument('--downsample_mode', type=str, default='none',
                        choices=['none', 'random_ratio', 'voxel', 'fps', 'voxel_fps'],
                        help='Point cloud downsampling mode: none, random_ratio, voxel, fps, or voxel_fps')
    parser.add_argument('--downsample_ratio', type=float, default=0.4,
                        help='Ratio of points to keep for random_ratio mode (0.0-1.0)')
    parser.add_argument('--downsample_voxel_size', type=float, default=None,
                        help='Voxel size for voxel/voxel_fps mode (None for auto from median NN distance)')
    parser.add_argument('--downsample_voxel_size_multiplier', type=float, default=2.0,
                        help='Multiplier for auto voxel size (used when downsample_voxel_size is None)')
    parser.add_argument('--downsample_target_points', type=int, default=None,
                        help='Target number of points for fps/voxel_fps mode')
    parser.add_argument('--downsample_recompute_normals', action='store_true', default=True,
                        help='Recompute normals after downsampling')
    parser.add_argument('--downsample_random_seed', type=int, default=42,
                        help='Random seed for reproducible downsampling')

    # Random plane cut augmentation (Real3D only - simulates partial scans)
    parser.add_argument('--plane_cut_enabled', action='store_true', default=True,
                        help='Enable random plane cut augmentation for Real3D (simulates partial scans)')
    parser.add_argument('--no_plane_cut', action='store_false', dest='plane_cut_enabled',
                        help='Disable random plane cut augmentation')
    parser.add_argument('--plane_cut_prob', type=float, default=0.7,
                        help='Probability of applying plane cut (0.0-1.0, rest keeps full object)')
    parser.add_argument('--plane_cut_r_min', type=float, default=0.3,
                        help='Minimum retention ratio for plane cut')
    parser.add_argument('--plane_cut_r_max', type=float, default=0.9,
                        help='Maximum retention ratio for plane cut')
    parser.add_argument('--plane_cut_horizontal_prob', type=float, default=0.7,
                        help='Probability of horizontal cut (vs diverse direction)')
    parser.add_argument('--plane_cut_horizontal_angle_max', type=float, default=20.0,
                        help='Max angle (degrees) from Z axis for horizontal cuts')
    parser.add_argument('--plane_cut_min_points', type=int, default=1024,
                        help='Minimum points to retain after cut')
    parser.add_argument('--plane_cut_coarse_edge_prob', type=float, default=0.3,
                        help='Probability of applying coarse/jagged edges to cuts (0.0-1.0)')
    parser.add_argument('--plane_cut_coarse_edge_noise', type=float, default=0.05,
                        help='Noise magnitude for coarse edges (relative to object size, 0.0-0.2 typical)')

    # Random edge segment cutout augmentation (removes edge segments like cutout in images)
    parser.add_argument('--edge_cutout_enabled', action='store_true', default=False,
                        help='Enable random edge segment cutout augmentation (removes edge segments)')
    parser.add_argument('--edge_cutout_prob', type=float, default=0.5,
                        help='Probability of applying edge segment cutout (0.0-1.0)')
    parser.add_argument('--edge_cutout_max_segments', type=int, default=6,
                        help='Maximum number of edge segments to remove (1 to max_segments)')
    parser.add_argument('--edge_cutout_k_neighbors', type=int, default=8,
                        help='Number of neighbors for KNN distance computation in edge detection')
    parser.add_argument('--edge_cutout_threshold_percentile', type=float, default=85.0,
                        help='Percentile threshold for identifying edge segments (higher = fewer edges)')
    parser.add_argument('--edge_cutout_min_points', type=int, default=1024,
                        help='Minimum points to retain after edge cutout')


    # #Adjust learning rate
    parser.add_argument('--lr', default=None, type=float,
                        help='base learning rate for schedulers; defaults to lr_D if omitted')
    parser.add_argument('--lr_G', default=2e-5, type=float, help='learning rate for generator')
    parser.add_argument('--lr_D', default=8e-5, type=float, help='learning rate for discriminator')
    parser.add_argument('--lr_schedule', type=str, default='cosine_after_step',
                        choices=['cosine_after_step', 'cosine_warmup', 'cosine_warmup_restarts'],
                        help='Learning rate schedule for discriminator updates')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs for cosine_warmup schedule')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine schedules')
    parser.add_argument('--first_cycle_steps', type=int, default=200,
                        help='First cycle length for cosine_warmup_restarts schedule')
    parser.add_argument('--cycle_mult', type=float, default=1.0,
                        help='Cycle length multiplier for cosine_warmup_restarts schedule')
    parser.add_argument('--max_lr', type=float, default=None,
                        help='Maximum learning rate for cosine_warmup_restarts schedule')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps inside each cosine_warmup_restarts cycle')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Cycle max LR decay factor for cosine_warmup_restarts schedule')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer: Adam, SGD, AdamW, RMSprop, Adagrad, Adadelta, Adamax, ASGD')
    parser.add_argument('--generator_optimizer', type=str, default='AdamW',
                        help='Generator optimizer: Adam, SGD, AdamW, RMSprop, Adagrad, Adamax, ASGD')
    parser.add_argument('--generator_beta1', type=float, default=0.9,
                        help='beta1 for generator optimizer (Adam/AdamW)')
    parser.add_argument('--generator_beta2', type=float, default=0.99,
                        help='beta2 for generator optimizer (Adam/AdamW)')
    parser.add_argument('--discriminator_beta1', type=float, default=0.9,
                        help='beta1 for discriminator optimizer (Adam/AdamW)')
    parser.add_argument('--discriminator_beta2', type=float, default=0.99,
                        help='beta2 for discriminator optimizer (Adam/AdamW)')
    parser.add_argument('--step_epoch', type=int, default=10,
                        help='How many steps apart to decay the learning rate')
    parser.add_argument('--multiplier', type=float, default=0.55,
                        help='Learning rate decay: lr = lr * multiplier')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float,
                        default=0.00015, help='weight_decay for SGD')

    # #model parameter
    parser.add_argument('--voxel_size', type=float,
                        default=0.03, help='voxel size')
    parser.add_argument('--in_channels', type=int,
                        default=3, help='in channels')
    parser.add_argument('--out_channels', type=int,
                        default=32, help='backbone feat channels')

    # Offset prediction head architecture (for ablation studies)
    parser.add_argument('--offset_head_variant', type=str, default='baseline',
                        choices=['baseline', 'deep', 'residual', 'multi_head', 'attention'],
                        help='Architecture variant for offset prediction head. Options: '
                             'baseline (original 3-layer MLP), '
                             'deep (deeper MLP with more capacity), '
                             'residual (deep MLP with residual connections), '
                             'multi_head (separate heads for x,y,z), '
                             'attention (attention-weighted features)')
    parser.add_argument('--offset_hidden_dim', type=int, default=64,
                        help='Hidden dimension for offset prediction head (deep/residual/multi_head/attention variants)')
    parser.add_argument('--offset_num_layers', type=int, default=3,
                        help='Number of layers in offset prediction head (deep/residual/multi_head variants)')
    parser.add_argument('--offset_dropout', type=float, default=0.0,
                        help='Dropout probability in offset prediction head (0.0-0.5 recommended)')
    parser.add_argument('--offset_attention_reduction', type=int, default=4,
                        help='Reduction ratio for attention module in attention variant')

    # Anomaly smart configs
    parser.add_argument('--smart_anomaly', default=True,
                        action='store_true', help='Whether to use smart anomaly')
    parser.add_argument('--no_smart_anomaly', action='store_false', dest='smart_anomaly',
                        help='Disable smart anomaly synthesis (use original Norm-AS module)')
    parser.add_argument('--R_alpha', type=float, default=1,
                        help='Alpha value for R')
    parser.add_argument('--R_beta', type=float, default=1,
                        help='Beta value for R')
    parser.add_argument('--R_low_bound', type=float,
                        default=0.03, help='Lower bound for R')
    parser.add_argument('--R_up_bound', type=float,
                        default=0.25, help='Upper bound for R')
    parser.add_argument('--B_alpha', type=float, default=1,
                        help='Alpha value for B')
    parser.add_argument('--B_beta', type=float, default=1,
                        help='Beta value for B')
    parser.add_argument('--B_low_bound', type=float,
                        default=0.06, help='Lower bound for B')
    parser.add_argument('--B_up_bound', type=float,
                        default=0.12, help='Upper bound for B')
    parser.add_argument('--cosine_kernel_prob', type=float,
                        default=0.4, help='Probability of using cosine kernel')
    parser.add_argument('--gaussian_kernel_prob', type=float,
                        default=0.2, help='Probability of using gaussian kernel')
    parser.add_argument('--poly_kernel_prob', type=float,
                        default=0.3, help='Probability of using poly kernel')
    parser.add_argument('--hard_kernel_prob', type=float,
                        default=0.1, help='Probability of using hard kernel')
    parser.add_argument('--poly_q', type=float, default=4,
                        help='q value for poly kernel')
    parser.add_argument('--one_sided_prob', type=float,
                        default=1, help='Probability of using one-sided kernel')

    # Anomaly preset distribution weights (controls sampling probability for each preset type)
    # Default weights are uniform (1.0 each). Higher values increase selection probability.
    parser.add_argument('--preset_0_weight', type=float, default=1.0,
                        help='Weight for Type 1: Basic Bulge preset')
    parser.add_argument('--preset_1_weight', type=float, default=1.0,
                        help='Weight for Type 2: Basic Dent preset')
    parser.add_argument('--preset_2_weight', type=float, default=1.0,
                        help='Weight for Type 3: Ridge preset')
    parser.add_argument('--preset_3_weight', type=float, default=1.0,
                        help='Weight for Type 4: Trench preset')
    parser.add_argument('--preset_4_weight', type=float, default=1.0,
                        help='Weight for Type 5: Elliptic Patch/Flat Spot preset')
    parser.add_argument('--preset_5_weight', type=float, default=1.0,
                        help='Weight for Type 6: Skewed Impact Crater preset')
    parser.add_argument('--preset_6_weight', type=float, default=1.0,
                        help='Weight for Type 7: Shear U preset')
    parser.add_argument('--preset_7_weight', type=float, default=1.0,
                        help='Weight for Type 7b: Shear V preset')
    parser.add_argument('--preset_8_weight', type=float, default=1.0,
                        help='Weight for Type 8: Double-sided Ripple preset')
    parser.add_argument('--preset_9_weight', type=float, default=1.0,
                        help='Weight for Type 9: Micro Dimple Field Base preset')
    parser.add_argument('--preset_10_weight', type=float, default=1.0,
                        help='Weight for Type 10: Directional Drag/Stretch preset')

    parser.add_argument('--micro_dimple_count', type=int, default=5,
                        help='Number of micro dimples to apply for preset 9 (Micro Dimple Field). '
                             'Each dimple is placed at a different random center within the anomaly region.')

    parser.add_argument('--binary_anomaly_label', action='store_true', default=False,
                                help='Use binary labels (1.0) for points in anomalous regions instead of gradual falloff weights (0 to 1)')

    parser.add_argument('--intact_ratio', type=float, default=0.0,
                        help='Fraction of training samples to keep intact without anomaly (0.0-1.0, default 0.0)')
    
    parser.add_argument('--cache_clear_freq', type=int, default=50,
                        help='Clear dataset cache every N epochs to prevent memory buildup (0 to disable)')
    parser.add_argument('--gc_collect_freq', type=int, default=50,
                        help='Run garbage collection and CUDA cache clearing every N epochs (0 to disable)')
    parser.add_argument('--memory_report_freq', type=int, default=50,
                        help='Report system and GPU memory usage every N epochs (0 to disable)')

    parser.add_argument('--train_eval_freq', type=int, default=5,
                        help='evaluate model on dedicated batch of pseudo anomalous samples every N epochs to check overfitting (0 to disable)')
    parser.add_argument('--train_eval_batch_size', type=int, default=8,
                        help='batch size for dedicated training evaluation (checking overfitting on training data)')
    parser.add_argument('--train_eval_num_batches', type=int, default=5,
                        help='number of batches to use for training evaluation')
    parser.add_argument('--sample_export_freq', type=int, default=50,
                        help='export anomaly samples every N epochs (0 to disable)')
    parser.add_argument('--sample_export_max', type=int, default=5,
                        help='maximum number of samples to export each time')
    parser.add_argument('--metric_eval_freq', type=int, default=1,
                        help='compute and log training AUC metrics every N epochs (0 to disable)')
    parser.add_argument('--metric_max_points', type=int, default=100000,
                        help='maximum number of rollout points sampled per metrics evaluation (<=0 for all)')

    parser.add_argument('--sample_export_all', action='store_true', default=False,
                        help='export all samples instead of limiting to sample_export_max')
    parser.add_argument('--sample_export_annotated', action='store_true', default=False,
                        help='export samples with colored anomaly annotations showing anomaly regions')

    parser.add_argument('--resume_checkpoint', type=str, default='',
                        help='optional path to a checkpoint to resume training from')


    args = parser.parse_args()
    args.num_workers = args.num_works
    return args
