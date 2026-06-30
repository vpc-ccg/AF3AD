import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.r3dad.utils.config import cmd_from_config
from integrations.r3dad.utils.dataset import all_shapenetad_cates


def main(args):

    exp_name = Path(args.config).stem
    time_fix = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    cfg_cmd = cmd_from_config(args.config)

    if 'ShapeNetAD' in cfg_cmd:
        cates = all_shapenetad_cates
        dataset = 'shapenet-ad'
    else:
        raise NotImplementedError
    
    train_script = Path(__file__).with_name("train_ae.py")
    for cate in cates:
        cmd = (
            f"python {train_script} --category {cate} "
            f"--log_root logs_{dataset}/{exp_name}_{time_fix}_{args.tag}/ "
            f"--save_ply True"
            + cfg_cmd
        )
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()
    main(args)
